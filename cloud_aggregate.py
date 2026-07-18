"""cloud_aggregate.py -- CLW-HFL Part 4.

Takes Layer C's client population (Backdoor/Normal, binary `head_binary`)
and re-aggregates their per-round updates through a genuinely cryptographic
pipeline, producing checkpoints/cloud_global.pth:

    client delta -> weight by sample count -> CLIP (DP sensitivity bound)
    -> distributed Gaussian DP noise (split across clients so the sum has
       the intended aggregate noise level)
    -> Bonawitz-style pairwise secure-aggregation masking
    -> quantize + pack + Paillier-encrypt (common/paillier_utils.py)
    -> aggregator homomorphically SUMS ciphertexts (never sees a plaintext
       individual update)
    -> decrypt the SUM only (separate "key holder" role) -> unpack ->
       dequantize -> FedAdam server update (common/fedopt.py)

Each of secure_aggregation / differential_privacy / paillier is toggled
independently via config.yaml flags, so run_ablation.py (Part 5) can
produce the exact "FedAvg baseline -> +GradClip -> +SecAgg -> +DistDP ->
+PartialHE" sequence for Figure 7 by flipping these flags between runs of
this same script -- not five different hand-written variants.

IMPORTANT (Part 2, item 1): evaluate_global.py must load THIS script's
output (cloud_global.pth) for Layer C's slice of the global confusion
matrix, never layer_C_final.pth (that file is plain-FedAvg Layer C only,
kept as a separate reference point / ablation baseline artifact).

Example:
    python cloud_aggregate.py --seed 42
    python cloud_aggregate.py --seed 42 --rounds 5   # quick smoke test
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.label_utils import load_label_map, build_layer_remap, filter_and_remap, layer_num_classes, LAYER_HEAD_NAME
from common.model import build_model_for_layer
from common.partition import dirichlet_partition
from common.focal_loss import ClassBalancedFocalLoss
from common.federated_trainer import _local_train_one_client, evaluate_on_subset
from common.vector_utils import flatten_state_dict, unflatten_to_state_dict, split_trainable_and_bn_buffers, average_bn_buffers
from common.secure_agg import mask_update
from common.dp_accounting import RDPAccountant
from common.fedopt import FedAdamState
from common.logging_utils import RoundLogger, timer
import common.paillier_utils as pu

LAYER = "C"  # Cloud aggregation operates on Layer C's client population (Part 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config["runtime"]["device"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sa_enabled = config["secure_aggregation"]["enabled"]
    dp_enabled = config["differential_privacy"]["enabled"]
    phe_enabled = config["paillier"]["enabled"]

    processed_dir = config["paths"]["processed_dir"]
    label_map = load_label_map(config["paths"]["label_map_path"])
    train_npz = np.load(os.path.join(processed_dir, "train.npz"))
    val_npz = np.load(os.path.join(processed_dir, "val.npz"))

    head_name = LAYER_HEAD_NAME[LAYER]
    num_classes = layer_num_classes(LAYER)
    global_to_local, local_to_global = build_layer_remap(LAYER, label_map)

    x_train, y_train = filter_and_remap(LAYER, label_map, train_npz["X"], train_npz["y"])
    x_val, y_val = filter_and_remap(LAYER, label_map, val_npz["X"], val_npz["y"])
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    num_clients = config["clients"][f"layer_{LAYER}"]
    alpha = config["partitioning"]["default_alpha"] if args.alpha is None else args.alpha
    client_indices = dirichlet_partition(y_train, num_clients, alpha, seed=args.seed)

    class_counts = np.bincount(y_train, minlength=num_classes)
    loss_fn = ClassBalancedFocalLoss(
        class_counts,
        beta=config["training"]["focal_loss"]["beta"],
        gamma=config["training"]["focal_loss"]["gamma"],
    )

    model = build_model_for_layer(
        LAYER, input_dim=config["model"]["input_dim"], hidden_dims=tuple(config["model"]["hidden_dims"]),
    ).to(device)

    num_rounds = args.rounds or config["training"]["rounds"]["cloud"]
    batch_size = config["training"]["batch_size"]
    local_epochs = config["training"]["local_epochs"]
    grad_clip_norm = config["training"]["grad_clip_norm"]
    optimizer_cfg = config["training"]["optimizer"]

    dp_clip_norm = config["differential_privacy"]["clip_norm"]
    noise_multiplier = config["differential_privacy"]["noise_multiplier"]
    target_delta = config["differential_privacy"]["target_delta"]

    checkpoint_dir = config["paths"]["checkpoints_dir"]
    log_dir = config["paths"]["logs_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- One-time setup: FedAdam state, RDP accountant, Paillier keypair ---
    # IMPORTANT: split BatchNorm running stats from learnable params.
    # Running stats (running_mean, running_var, num_batches_tracked) must NOT
    # go through the FedAdam+quantization+PHE pipeline -- that pipeline is
    # designed for learnable parameter deltas, not accumulated statistics.
    # Including them causes eval-mode BatchNorm to use corrupted running stats,
    # which makes the model always output the majority class in eval mode.
    # Fix: only FedAdam+PHE the trainable params; average the BN buffers
    # separately via plain weighted average (they don't need encryption --
    # they're descriptive statistics of local data distributions, not private
    # gradient information in the DP threat model sense).
    _init_trainable, _init_bn = split_trainable_and_bn_buffers(model.active_state_dict(head_name))
    dim = len(flatten_state_dict(_init_trainable)[0])
    fedadam = FedAdamState(
        dim,
        server_lr=config["fedopt"]["server_lr"],
        beta1=config["fedopt"]["beta1"],
        beta2=config["fedopt"]["beta2"],
        tau=config["fedopt"]["tau"],
    )
    rdp_accountant = RDPAccountant(target_delta=target_delta) if dp_enabled else None

    # NOTE on master_secret (Part 4 / common/secure_agg.py): derived
    # deterministically from --seed so results are reproducible across
    # re-runs for the paper. This is a research-simulation convenience, not
    # a production secrecy property -- in a real deployment each pairwise
    # secret would come from a genuine Diffie-Hellman key agreement between
    # clients, invisible to the server, regardless of any public seed.
    master_secret = hashlib.sha256(f"clw-hfl-cloud-master-secret-seed-{args.seed}".encode()).digest()
    # mask_scale must be small enough that even the WORST-CASE total mask a
    # client could accumulate (all (num_clients-1) pairwise terms aligned in
    # sign, which is astronomically unlikely but must still be guaranteed
    # safe) plus the largest possible clipped-delta contribution never
    # exceeds quant_clip -- otherwise quantize()'s clipping would silently
    # truncate the mask itself and corrupt the exact cancellation property
    # common/secure_agg.py relies on. Derived with a 2x safety factor.
    quant_clip = config["paillier"]["quant_clip"]
    worst_case_mask_multiplier = max(1, num_clients - 1)
    mask_scale = (quant_clip - dp_clip_norm) / worst_case_mask_multiplier / 2.0
    assert mask_scale > 0, "quant_clip must exceed differential_privacy.clip_norm"

    paillier_pub, paillier_priv = None, None
    key_gen_time = None
    if phe_enabled:
        from phe import paillier

        pu.validate_packing_params(
            key_size_bits=config["paillier"]["key_size_bits"],
            slot_bits=config["paillier"]["slot_bits"],
            pack_batch_size=config["paillier"]["pack_batch_size"],
            max_clients=num_clients,
            quant_scale=config["paillier"]["quant_scale"],
            quant_clip=config["paillier"]["quant_clip"],
        )
        with timer() as t:
            paillier_pub, paillier_priv = paillier.generate_paillier_keypair(
                n_length=config["paillier"]["key_size_bits"]
            )
        key_gen_time = t.elapsed
        print(f"[cloud] generated {config['paillier']['key_size_bits']}-bit Paillier keypair "
              f"in {key_gen_time:.2f}s")

    log_path = os.path.join(log_dir, f"cloud_alpha{alpha}_seed{args.seed}_rounds.jsonl")
    logger = RoundLogger(log_path)
    rng = np.random.RandomState(args.seed)
    training_start = time.perf_counter()

    print(f"[cloud] {num_clients} clients, {num_rounds} rounds, alpha={alpha}, seed={args.seed} | "
          f"secure_agg={sa_enabled} dp={dp_enabled} phe={phe_enabled}")

    for round_idx in range(num_rounds):
        participant_ids = list(range(num_clients))
        # Split: only trainable params go through FedAdam+PHE pipeline;
        # BN running stats are averaged separately (see dim setup above).
        global_trainable, global_bn = split_trainable_and_bn_buffers(model.active_state_dict(head_name))
        global_flat, layout = flatten_state_dict(global_trainable)

        client_payloads = []
        client_bn_buffers = []   # collected for plain weighted average
        client_records = []
        total_samples = sum(len(client_indices[c]) for c in participant_ids if len(client_indices[c]) > 0)
        active_participants = [c for c in participant_ids if len(client_indices[c]) > 0]

        encrypt_time_total = 0.0
        for ci, cid in enumerate(active_participants):
            print(f"[cloud]   round {round_idx + 1}: client {ci + 1}/{len(active_participants)} "
                  f"(id={cid}) training locally...", flush=True)
            idxs = client_indices[cid]
            local_model = copy.deepcopy(model)
            with timer() as t_compute:
                _local_train_one_client(
                    local_model, x_train_t[idxs], y_train_t[idxs],
                    optimizer_cfg, batch_size, local_epochs, grad_clip_norm,
                    head_name, loss_fn, device,
                )
            local_trainable, local_bn = split_trainable_and_bn_buffers(local_model.active_state_dict(head_name))
            client_bn_buffers.append(local_bn)

            local_flat, _ = flatten_state_dict(local_trainable)
            delta = local_flat - global_flat
            weight = len(idxs) / total_samples
            weighted_delta = delta * weight

            raw_norm = float(np.linalg.norm(weighted_delta))
            print(f"[cloud]   round {round_idx + 1}: client {ci + 1}/{len(active_participants)} "
                  f"raw weighted-delta L2 norm (pre-clip) = {raw_norm:.6f} "
                  f"(dp_clip_norm={dp_clip_norm})", flush=True)

            # --- DP sensitivity clipping ---
            norm = np.linalg.norm(weighted_delta)
            if norm > dp_clip_norm:
                weighted_delta = weighted_delta * (dp_clip_norm / norm)

            # --- Distributed DP noise: each client adds a SHARE of the
            # total intended noise, so the noise on the SUM across
            # n participants matches the standard Gaussian mechanism with
            # std = noise_multiplier * clip_norm (Part 4). ---
            if dp_enabled:
                individual_std = noise_multiplier * dp_clip_norm / np.sqrt(len(active_participants))
                weighted_delta = weighted_delta + rng.normal(0.0, individual_std, size=weighted_delta.shape)

            plaintext_bytes = weighted_delta.astype(np.float32).nbytes

            # --- Secure aggregation masking (Bonawitz-style pairwise) ---
            payload_vec = weighted_delta
            if sa_enabled:
                payload_vec = mask_update(master_secret, round_idx, cid, active_participants, payload_vec, mask_scale)

            # --- Partial Homomorphic Encryption (quantize+pack+encrypt) ---
            this_encrypt_time = 0.0
            transmitted_bytes = plaintext_bytes
            if phe_enabled:
                q, offset = pu.quantize(payload_vec, config["paillier"]["quant_scale"], config["paillier"]["quant_clip"])
                packed, true_len = pu.pack(q, config["paillier"]["pack_batch_size"], config["paillier"]["slot_bits"])
                print(f"[cloud]   round {round_idx + 1}: client {ci + 1}/{len(active_participants)} "
                      f"encrypting {len(packed)} ciphertexts (this can take a while without "
                      f"gmpy2 installed)...", flush=True)
                ciphertexts, this_encrypt_time = pu.encrypt_packed(packed, paillier_pub)
                print(f"[cloud]   round {round_idx + 1}: client {ci + 1}/{len(active_participants)} "
                      f"encryption done in {this_encrypt_time:.1f}s", flush=True)
                encrypt_time_total += this_encrypt_time
                transmitted_bytes = pu.ciphertext_bytes(ciphertexts)
                client_payloads.append(ciphertexts)
            else:
                client_payloads.append(payload_vec)

            client_records.append({
                "client_id": int(cid),
                "num_samples": int(len(idxs)),
                "compute_time_sec": t_compute.elapsed,
                "encrypt_time_sec": this_encrypt_time,
                "plaintext_message_bytes": int(plaintext_bytes),
                "transmitted_message_bytes": int(transmitted_bytes),
            })

        # --- Aggregation: sum across clients. The aggregator NEVER sees an
        # individual client's plaintext payload at this point -- only
        # ciphertexts (if PHE enabled) or already-masked vectors (if SecAgg
        # enabled without PHE). ---
        decrypt_time = 0.0
        if phe_enabled:
            print(f"[cloud]   round {round_idx + 1}: summing ciphertexts across "
                  f"{len(active_participants)} clients and decrypting...", flush=True)
            summed_cts = pu.sum_ciphertexts_across_clients(client_payloads)
            decrypted_ints, decrypt_time = pu.decrypt_summed(summed_cts, paillier_priv)
            print(f"[cloud]   round {round_idx + 1}: decryption done in {decrypt_time:.1f}s", flush=True)
            unpacked = pu.unpack(decrypted_ints, config["paillier"]["pack_batch_size"],
                                  config["paillier"]["slot_bits"], true_len)
            recovered_sum = pu.dequantize_sum(unpacked, len(active_participants), offset,
                                               config["paillier"]["quant_scale"])
        else:
            recovered_sum = np.sum(client_payloads, axis=0)
        # If SecAgg masks were applied, they cancel out exactly in this sum
        # by construction (common/secure_agg.py) -- recovered_sum already
        # equals sum_i(clipped[+noised] weighted_delta_i) either way.
        print(f"[cloud]   round {round_idx + 1}: recovered aggregate vector L2 norm = "
              f"{float(np.linalg.norm(recovered_sum)):.6f} "
              f"(expected DP noise std per-dim ~{(noise_multiplier * dp_clip_norm) if dp_enabled else 0.0}, "
              f"dim={len(recovered_sum)} -> expected noise-only norm ~"
              f"{(noise_multiplier * dp_clip_norm * np.sqrt(len(recovered_sum))) if dp_enabled else 0.0:.2f})",
              flush=True)

        new_global_flat = fedadam.step(global_flat, recovered_sum)
        # Rebuild the full state dict: FedAdam-updated trainable params + plain-averaged BN buffers
        client_weights_for_bn = [len(client_indices[c]) for c in active_participants]
        new_bn = average_bn_buffers(client_bn_buffers, client_weights_for_bn)
        dtypes = {k: v.dtype for k, v in global_trainable.items()}
        new_trainable_state = unflatten_to_state_dict(new_global_flat, layout, dtypes=dtypes)
        new_active_state = {**new_trainable_state, **new_bn}
        model.load_state_dict(new_active_state, strict=False)

        val_metrics = evaluate_on_subset(model, x_val_t, y_val_t, head_name, num_classes, device)

        cumulative_epsilon = None
        if dp_enabled:
            cumulative_epsilon = rdp_accountant.step(round_idx, noise_multiplier)

        record = {
            "round": round_idx,
            "alpha": alpha,
            "seed": args.seed,
            "secure_aggregation_enabled": sa_enabled,
            "dp_enabled": dp_enabled,
            "phe_enabled": phe_enabled,
            "num_clients_this_round": len(active_participants),
            "clients": client_records,
            "total_plaintext_bytes_this_round": sum(c["plaintext_message_bytes"] for c in client_records),
            "total_transmitted_bytes_this_round": sum(c["transmitted_message_bytes"] for c in client_records),
            "total_encrypt_time_sec": encrypt_time_total,
            "decrypt_time_sec": decrypt_time,
            "dp_noise_sigma": (noise_multiplier * dp_clip_norm) if dp_enabled else None,
            "cumulative_epsilon": cumulative_epsilon,
            "target_delta": target_delta if dp_enabled else None,
            "val_accuracy": val_metrics["accuracy"] if val_metrics else None,
            "val_f1_macro": val_metrics["f1_macro"] if val_metrics else None,
        }
        logger.log(record)

        eps_str = f"{cumulative_epsilon:.2f}" if cumulative_epsilon is not None else "n/a"
        print(f"[cloud] round {round_idx + 1}/{num_rounds} "
              f"val_acc={record['val_accuracy']:.4f} val_f1_macro={record['val_f1_macro']:.4f} "
              f"cumulative_eps={eps_str} "
              f"encrypt={encrypt_time_total:.2f}s decrypt={decrypt_time:.2f}s")

    total_training_time = time.perf_counter() - training_start
    logger.to_csv(os.path.join(log_dir, f"cloud_alpha{alpha}_seed{args.seed}_rounds.csv"))

    final_val_metrics = evaluate_on_subset(model, x_val_t, y_val_t, head_name, num_classes, device)

    ckpt_path = os.path.join(checkpoint_dir, "cloud_global.pth")
    torch.save(model.state_dict(), ckpt_path)

    metadata = {
        "layer": LAYER,
        "head_name": head_name,
        "classes": list(global_to_local.keys()),
        "global_to_local": {int(k): int(v) for k, v in global_to_local.items()},
        "local_to_global": {int(k): int(v) for k, v in local_to_global.items()},
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": args.seed,
        "num_rounds": num_rounds,
        "secure_aggregation_enabled": sa_enabled,
        "dp_enabled": dp_enabled,
        "phe_enabled": phe_enabled,
        "paillier_key_gen_time_sec": key_gen_time,
        "total_training_time_sec": total_training_time,
        "final_val_metrics": final_val_metrics,
        "final_privacy": rdp_accountant.summary() if dp_enabled else None,
    }
    meta_path = os.path.join(checkpoint_dir, "cloud_global_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[cloud] done in {total_training_time:.1f}s -> {ckpt_path}")
    print(f"[cloud] final val_accuracy={final_val_metrics['accuracy']:.4f} "
          f"val_f1_macro={final_val_metrics['f1_macro']:.4f}")
    if dp_enabled:
        print(f"[cloud] final cumulative privacy: {rdp_accountant.summary()}")


if __name__ == "__main__":
    main()