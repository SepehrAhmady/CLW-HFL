"""Quantize + pack + Paillier-encrypt pipeline for tractable PHE aggregation.

Per Part 1, section 0: real homomorphic encryption applied naively to every
individual float would be impractically slow. This module implements the
"quantize to fixed-point, pack several scalars into one ciphertext" scheme
that Part 1 prescribes:

  1. QUANTIZE: each float is clamped to [-quant_clip, quant_clip] and
     scaled to a fixed-point integer (multiply by quant_scale, round).
  2. OFFSET: add a constant OFFSET = round(quant_clip * quant_scale) so
     every quantized value becomes non-negative (Paillier's plaintext
     space is non-negative integers mod n; this also makes the *sum*
     across multiple clients still land in a clean, boundable range).
  3. PACK: pack `pack_batch_size` non-negative quantized scalars into ONE
     big integer using base 2**slot_bits (positional/SIMD packing), so one
     Paillier ciphertext carries many scalars -- this is what makes
     encryption tractable (Part 1, section 0) instead of one ciphertext
     per float.
  4. ENCRYPT each client's packed big integers with the (shared) Paillier
     public key.
  5. HOMOMORPHICALLY SUM ciphertexts across clients for each batch index
     (Paillier's defining property: E(a) + E(b) == E(a+b); the "+" here is
     EncryptedNumber addition from the `phe` library). The aggregator only
     ever touches ciphertexts at this stage.
  6. DECRYPT the summed ciphertexts (only the party holding the private
     key can do this -- see cloud_aggregate.py for how that role is kept
     conceptually separate from the summing/aggregation code path).
  7. UNPACK each decrypted big integer back into its `pack_batch_size`
     slots, subtract n_clients * OFFSET (since OFFSET was added once per
     client, summing n_clients of them adds n_clients * OFFSET), and
     divide by quant_scale to recover the true (summed) float values.

CRITICAL SIZING CONSTRAINT (validate_packing_params): slot_bits must be
large enough that no slot's value, after being doubled in range (offset)
and summed across up to `max_clients` clients, can overflow into the
neighbouring slot. And pack_batch_size * slot_bits must stay safely below
the Paillier key size, or the packed integer doesn't fit in the plaintext
space at all. Both are asserted explicitly here rather than silently
producing corrupted results -- config.yaml's paillier section documents
the derivation for the shipped defaults (2048-bit key, 32 slot_bits, 64
batch size).
"""
import math

import numpy as np


def validate_packing_params(key_size_bits, slot_bits, pack_batch_size,
                             max_clients, quant_scale, quant_clip, margin_bits=24):
    """Raises AssertionError with a clear message if the configured
    sizing is not internally consistent. Call this once at startup.
    """
    offset = round(quant_clip * quant_scale)
    max_slot_value_after_sum = max_clients * 2 * offset
    bits_needed = max_slot_value_after_sum.bit_length()
    assert slot_bits >= bits_needed, (
        f"slot_bits={slot_bits} is too small: with quant_clip={quant_clip}, "
        f"quant_scale={quant_scale}, max_clients={max_clients}, each slot can "
        f"reach {max_slot_value_after_sum} after summation, which needs "
        f"{bits_needed} bits. Increase slot_bits or reduce quant_clip/quant_scale."
    )
    total_bits = slot_bits * pack_batch_size
    assert total_bits + margin_bits < key_size_bits, (
        f"pack_batch_size={pack_batch_size} * slot_bits={slot_bits} = "
        f"{total_bits} bits, which does not fit safely under a "
        f"key_size_bits={key_size_bits} Paillier modulus (margin={margin_bits}). "
        f"Reduce pack_batch_size or slot_bits, or increase key_size_bits."
    )


def quantize(values: np.ndarray, quant_scale: float, quant_clip: float):
    """Returns (q_nonneg: np.ndarray[int64], offset: int).
    q_nonneg[i] = round(clip(values[i], -quant_clip, quant_clip) * quant_scale) + offset
    """
    offset = round(quant_clip * quant_scale)
    clipped = np.clip(values, -quant_clip, quant_clip)
    q = np.round(clipped * quant_scale).astype(np.int64)
    return q + offset, offset


def dequantize_sum(summed_q_nonneg: np.ndarray, n_clients: int, offset: int, quant_scale: float):
    """Inverse of quantize(), applied to a value that is the SUM of
    n_clients individually-quantized-and-offset vectors.
    """
    return (summed_q_nonneg.astype(np.float64) - n_clients * offset) / quant_scale


def pack(q_nonneg: np.ndarray, pack_batch_size: int, slot_bits: int):
    """Pack a 1D non-negative int64 array into a list of Python big ints,
    `pack_batch_size` scalars per int, using base 2**slot_bits.
    Returns (packed_ints: list[int], padded_length: int) -- padded_length
    records how many scalars the LAST batch was padded with zeros to reach
    pack_batch_size, so unpack() can trim them back off.
    """
    n = len(q_nonneg)
    n_batches = math.ceil(n / pack_batch_size)
    padded_len = n_batches * pack_batch_size
    padded = np.zeros(padded_len, dtype=object)  # python int dtype to avoid overflow
    padded[:n] = [int(v) for v in q_nonneg]

    base = 1 << slot_bits
    packed_ints = []
    for b in range(n_batches):
        chunk = padded[b * pack_batch_size:(b + 1) * pack_batch_size]
        big = 0
        for k in range(pack_batch_size - 1, -1, -1):
            big = big * base + int(chunk[k])
        # NOTE: building positionally so slot k occupies base**k -- equivalent
        # but built via Horner's method (high-to-low) for efficiency/clarity.
        packed_ints.append(big)
    return packed_ints, n


def unpack(packed_ints, pack_batch_size: int, slot_bits: int, true_length: int):
    """Inverse of pack(). Returns a 1D np.ndarray[object] (python ints) of
    length true_length (padding from the last batch is trimmed off).
    """
    base = 1 << slot_bits
    out = []
    for big in packed_ints:
        chunk = []
        remaining = big
        for _ in range(pack_batch_size):
            chunk.append(remaining % base)
            remaining //= base
        out.extend(chunk)
    return np.array(out[:true_length], dtype=object)


# --- Actual Paillier (phe library) wrapping --------------------------------
# Everything above this point is pure integer arithmetic and is fully
# exercised by tests using plain Python ints (Paillier's homomorphic
# addition is BY DEFINITION equivalent to plaintext integer addition, so
# those tests already validate the arithmetic this section wraps). The
# functions below make the actual `phe.paillier` library calls.

def encrypt_packed(packed_ints, public_key):
    """Encrypts each packed big-int with the shared Paillier public key.
    Returns (ciphertexts: list[EncryptedNumber], encrypt_time_sec: float).
    """
    import time
    from phe import paillier  # noqa: F401  (imported for clarity/availability check)

    start = time.perf_counter()
    ciphertexts = [public_key.encrypt(int(v)) for v in packed_ints]
    elapsed = time.perf_counter() - start
    return ciphertexts, elapsed


def sum_ciphertexts_across_clients(per_client_ciphertexts):
    """per_client_ciphertexts: list (one per client) of list (one per batch)
    of EncryptedNumber. Returns one list of EncryptedNumber, one per batch,
    each being the homomorphic sum across all clients for that batch index.
    This is the ONLY operation the aggregator performs on ciphertexts --
    it never decrypts and never sees an individual client's plaintext.
    """
    n_batches = len(per_client_ciphertexts[0])
    summed = []
    for b in range(n_batches):
        acc = per_client_ciphertexts[0][b]
        for client_cts in per_client_ciphertexts[1:]:
            acc = acc + client_cts[b]  # EncryptedNumber.__add__ -> homomorphic addition
        summed.append(acc)
    return summed


def decrypt_summed(summed_ciphertexts, private_key):
    """Decrypts the already-summed ciphertexts (one per batch) using the
    private key. Returns (plaintext_ints: list[int], decrypt_time_sec: float).
    Only the party holding `private_key` can call this -- in
    cloud_aggregate.py this is kept as a logically separate role from the
    ciphertext-summing aggregator, even though both run in the same process
    for this single-machine simulation.
    """
    import time

    start = time.perf_counter()
    plaintexts = [private_key.decrypt(ct) for ct in summed_ciphertexts]
    elapsed = time.perf_counter() - start
    return plaintexts, elapsed


def ciphertext_bytes(ciphertexts):
    """Real serialized size (bytes) of a list of EncryptedNumber, used as
    the "encrypted" side of the Figure 6 message-size comparison. Each
    EncryptedNumber's ciphertext is a big int; .bit_length() converted to
    bytes is its exact wire size (ignoring negligible protocol overhead).
    """
    return sum((ct.ciphertext().bit_length() + 7) // 8 for ct in ciphertexts)
