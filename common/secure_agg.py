"""Bonawitz et al.-style pairwise additive secret masking.

Part 4 requires a REAL secure aggregation protocol -- not plain FedAvg
renamed (Part 2, item 4). The construction here is the standard pairwise
additive-mask scheme: for every unordered pair of participating clients
{i, j} with i < j, both clients deterministically derive the SAME random
mask vector r_ij from a shared seed, but apply it with opposite sign:

    client i adds  +r_ij to its update for every j > i
    client i adds  -r_ji to its update for every j < i   (using r_ij = r_ji)

Equivalently, client i's total mask is:
    mask_i = sum_{j > i} r(i,j)  -  sum_{j < i} r(j,i)

When all masked updates are summed across every participating client, every
pairwise mask appears exactly twice with opposite sign and telescopes to
exactly zero:
    sum_i mask_i = sum_{i<j} ( r(i,j) - r(i,j) ) = 0

So the AGGREGATOR, which only ever sees each client's (update + mask_i),
recovers sum_i update_i exactly once it sums across all clients -- without
ever seeing an individual client's mask_i or unmasked update_i in
isolation. In a real deployment the shared per-pair seed would come from a
Diffie-Hellman key agreement between each pair of clients (as in the
original Bonawitz et al. 2017 protocol); here, since this is a
single-machine research simulation, the "shared secret" is simulated by a
deterministic seed derived from (round_id, min(i,j), max(i,j), a private
master_secret) that is only ever used inside per-client mask-generation
calls, never exposed to the aggregation/summing code path -- see
cloud_aggregate.py for how the boundary between "client-side" and
"server-side" code is kept separate even within one process.
"""
import hashlib

import numpy as np


def _pairwise_seed(master_secret: bytes, round_id: int, i: int, j: int) -> int:
    """Deterministic 64-bit seed shared by clients i and j for this round,
    derived from a master secret neither the aggregator nor any other
    client pair has access to. lo, hi convention makes this symmetric in
    (i, j) so both clients independently derive the identical seed.
    """
    lo, hi = min(i, j), max(i, j)
    payload = master_secret + round_id.to_bytes(8, "big") + lo.to_bytes(4, "big") + hi.to_bytes(4, "big")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _pairwise_mask_vector(master_secret: bytes, round_id: int, i: int, j: int,
                           dim: int, scale: float) -> np.ndarray:
    seed = _pairwise_seed(master_secret, round_id, i, j)
    rng = np.random.RandomState(seed % (2**32 - 1))
    return rng.uniform(-scale, scale, size=dim)


def client_mask(master_secret: bytes, round_id: int, client_id: int,
                participant_ids, dim: int, scale: float) -> np.ndarray:
    """The mask a single client adds to its own update before transmission.
    `participant_ids` is the full list of client ids participating in this
    round (must be the identical set for every client's call, so masks
    telescope correctly across the whole round).
    """
    mask = np.zeros(dim, dtype=np.float64)
    for j in participant_ids:
        if j == client_id:
            continue
        r = _pairwise_mask_vector(master_secret, round_id, client_id, j, dim, scale)
        mask += r if j > client_id else -r
    return mask


def mask_update(master_secret: bytes, round_id: int, client_id: int,
                 participant_ids, update: np.ndarray, scale: float) -> np.ndarray:
    """What client `client_id` actually transmits: its true update plus its
    own pairwise mask. The aggregator only ever sees this masked value.
    """
    return update + client_mask(master_secret, round_id, client_id, participant_ids, len(update), scale)
