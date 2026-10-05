"""hfl_no_task_decomposition.py -- HFL *without* Task Decomposition (ablation).

Same hierarchy and same cloud privacy stack as CLW-HFL, but the ten attack
classes are NOT split across layers. Every tier (Edge / Router / Gateway)
trains the SAME full 10-class model (`head_fine`) on clients that hold ALL
classes, so the only thing removed relative to CLW-HFL is the
class-to-layer assignment (LAYER_ATTACK_MAP) and the oracle routing it needs.

Pipeline per round:

    global model (10-class)
      -> Tier A (30 clients) / Tier B (10 clients) / Tier C (10 clients):
           each client trains locally (Class-Balanced Focal Loss + grad clip),
           tier aggregator does weighted FedAvg over its own clients
      -> each tier sends ONE update (tier_model - global) to the cloud
      -> cloud: weight by tier sample count -> DP clip -> distributed
         Gaussian DP noise -> pairwise SecAgg masking -> quantize+pack+
         Paillier -> homomorphic SUM -> decrypt SUM only -> FedAdam
      -> BatchNorm running stats are averaged separately (plain weighted avg),
         exactly as in cloud_aggregate.py

The privacy stack is driven by the same config.yaml flags as cloud_aggregate.py
(secure_aggregation / differential_privacy / paillier .enabled). `--plain`
switches all three off for a "hierarchy only" variant, and `--server-opt fedavg`
replaces FedAdam with a plain weighted-delta server step.

Evaluation is a direct 10-class argmax on the test split (no routing), the
same protocol as flat_fedavg_baseline.py, so the numbers are comparable.

Outputs:
    logs/hfl_no_td_seed<s>_rounds.jsonl / .csv   (per-round log)
    checkpoints/hfl_no_td_seed<s>.pth            (final global model)
    logs/hfl_no_td_results.json                  (per-seed + mean +/- std)

Example:
    python hfl_no_task_decomposition.py
    python hfl_no_task_decomposition.py --seeds 42 --rounds 5     # smoke test
    python hfl_no_task_decomposition.py --plain                   # no SA/DP/PHE
    python hfl_no_task_decomposition.py --plain --server-opt fedavg   # == flat FedAvg
    python hfl_no_task_decomposition.py --no-dp --no-phe          # SecAgg + FedAdam only

    python hfl_no_task_decomposition.py --plain --edge-rounds 2 2 1   # budget-matched vs CLW-HFL

Note: with full participation, plain two-level weighted FedAvg is mathematically
identical to flat FedAvg, so `--plain --server-opt fedavg` should reproduce
flat_fedavg_baseline.py up to RNG/sampling differences.
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

from common.label_utils import load_label_map, CANONICAL_LABEL_MAP, NUM_GLOBAL_CLASSES
from common.model import MultiHeadIDSMLP
from common.focal_loss import ClassBalancedFocalLoss
from common.metrics import aggregate_seeds
from common.partition import dirichlet_partition
from common.federated_trainer import _local_train_one_client, evaluate_on_subset
from common.vector_utils import (
    flatten_state_dict, unflatten_to_state_dict,
    split_trainable_and_bn_buffers, average_bn_buffers,
)
from common.secure_agg import mask_update
from common.dp_accounting import RDPAccountant
from common.fedopt import FedAdamState
from common.logging_utils import RoundLogger, state_dict_size_bytes, timer
import common.paillier_utils as pu

HEAD = "head_fine"                      # the 10-class head, used by every tier
TIERS = ["A", "B", "C"]                 # Edge / Router / Gateway (same sizes as CLW-HFL)


def tier_client_ranges(config):
    """Contiguous client-id blocks per tier, e.g. A=[0,30) B=[30,40) C=[40,50)."""
    ranges, start = {}, 0
    for t in TIERS:
        n = config["clients"][f"layer_{t}"]
        ranges[t] = list(range(start, start + n))
        start += n
    return ranges, start


def run_one_seed(config, seed, alpha, num_rounds, device, data, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t = data
    cfg_t = config["training"]
    batch_size, local_epochs = cfg_t["batch_size"], cfg_t["local_epochs"]
    grad_clip_norm, opt_cfg = cfg_t["grad_clip_norm"], cfg_t["optimizer"]
    client_fraction = config["clients"].get("client_fraction_per_round", 1.0)

    sa_enabled = config["secure_aggregation"]["enabled"] and not (args.plain or args.no_secagg)
    dp_enabled = config["differential_privacy"]["enabled"] and not (args.plain or args.no_dp)
    phe_enabled = config["paillier"]["enabled"] and not (args.plain or args.no_phe)
    dp_clip = config["differential_privacy"]["clip_norm"]
    noise_mult = (config["differential_privacy"]["noise_multiplier"]
                  if args.noise_multiplier is None else args.noise_multiplier)
    target_delta = config["differential_privacy"]["target_delta"]
    pcfg = config["paillier"]

    # ---- data: ALL classes partitioned over ALL clients, then grouped by tier ----
    tier_clients, total_clients = tier_client_ranges(config)
    y_np = y_train_t.numpy()
    client_indices = dirichlet_partition(y_np, total_clients, alpha, seed=seed)

    class_counts = np.bincount(y_np, minlength=NUM_GLOBAL_CLASSES)
    loss_fn = ClassBalancedFocalLoss(
        class_counts, beta=cfg_t["focal_loss"]["beta"], gamma=cfg_t["focal_loss"]["gamma"],
    )

    model = MultiHeadIDSMLP(
        input_dim=config["model"]["input_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        num_classes_fine=NUM_GLOBAL_CLASSES,
    ).to(device)

    # ---- cloud state (one-time) ----
    init_trainable, _ = split_trainable_and_bn_buffers(model.active_state_dict(HEAD))
    dim = len(flatten_state_dict(init_trainable)[0])
    fedadam = FedAdamState(
        dim, server_lr=config["fedopt"]["server_lr"], beta1=config["fedopt"]["beta1"],
        beta2=config["fedopt"]["beta2"], tau=config["fedopt"]["tau"],
    )
    rdp = RDPAccountant(target_delta=target_delta) if dp_enabled else None
    master_secret = hashlib.sha256(f"clw-hfl-nodecomp-master-secret-seed-{seed}".encode()).digest()

    n_tiers = len(TIERS)
    # Same safety derivation as cloud_aggregate.py, with the cloud's participants
    # being the tier aggregators (n_tiers) instead of Layer-C clients.
    mask_scale = (pcfg["quant_clip"] - dp_clip) / max(1, n_tiers - 1) / 2.0
    assert mask_scale > 0, "paillier.quant_clip must exceed differential_privacy.clip_norm"

    pub = priv = None
    key_gen_time = None
    if phe_enabled:
        from phe import paillier
        pu.validate_packing_params(
            key_size_bits=pcfg["key_size_bits"], slot_bits=pcfg["slot_bits"],
            pack_batch_size=pcfg["pack_batch_size"], max_clients=n_tiers,
            quant_scale=pcfg["quant_scale"], quant_clip=pcfg["quant_clip"],
        )
        with timer() as t:
            pub, priv = paillier.generate_paillier_keypair(n_length=pcfg["key_size_bits"])
        key_gen_time = t.elapsed
        print(f"[hfl-no-td] seed={seed}: {pcfg['key_size_bits']}-bit Paillier keypair in {key_gen_time:.2f}s")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger = RoundLogger(os.path.join(args.log_dir, f"hfl_no_td_seed{seed}_rounds.jsonl"))

    print(f"[hfl-no-td] seed={seed} alpha={alpha} rounds={num_rounds} "
          f"tiers={ {t: len(c) for t, c in tier_clients.items()} } | "
          f"edge_rounds={dict(zip(TIERS, args.edge_rounds))} | "
          f"secagg={sa_enabled} dp={dp_enabled} phe={phe_enabled} server={args.server_opt}", flush=True)

    start = time.perf_counter()
    for round_idx in range(num_rounds):
        global_trainable, _ = split_trainable_and_bn_buffers(model.active_state_dict(HEAD))
        global_flat, layout = flatten_state_dict(global_trainable)
        dtypes = {k: v.dtype for k, v in global_trainable.items()}

        # ---------- Tier level: R_t edge rounds of local training + intra-tier FedAvg ----------
        # With edge_rounds == (1,1,1) this is mathematically identical to flat FedAvg.
        # With edge_rounds > 1 a tier refines its own model several times before it
        # talks to the cloud (real hierarchical FL; budget-matched vs CLW-HFL).
        tier_results = []   # one entry per ACTIVE tier
        for ti, tier in enumerate(TIERS):
            pool = tier_clients[tier]
            tier_model = copy.deepcopy(model)
            tier_flat, tier_bn = global_flat, None
            msg_bytes, compute_times, n_edge_done, tier_samples = 0, [], 0, 0

            for _e in range(args.edge_rounds[ti]):
                n_sel = max(1, int(round(client_fraction * len(pool))))
                chosen = rng.choice(pool, size=n_sel, replace=False)

                flats, bns, weights, round_compute = [], [], [], []
                for cid in chosen:
                    idxs = client_indices[cid]
                    if len(idxs) == 0:
                        continue
                    local = copy.deepcopy(tier_model)
                    with timer() as t:
                        _local_train_one_client(
                            local, x_train_t[idxs], y_train_t[idxs], opt_cfg, batch_size,
                            local_epochs, grad_clip_norm, HEAD, loss_fn, device,
                        )
                    active = local.active_state_dict(HEAD)
                    tr, bn = split_trainable_and_bn_buffers(active)
                    flats.append(flatten_state_dict(tr)[0])
                    bns.append(bn)
                    weights.append(len(idxs))
                    msg_bytes += state_dict_size_bytes(active)
                    round_compute.append(t.elapsed)

                if not flats:
                    continue
                w = np.asarray(weights, dtype=np.float64)
                tier_flat = np.sum([f * (wi / w.sum()) for f, wi in zip(flats, w)], axis=0)
                tier_bn = average_bn_buffers(bns, weights)
                tier_samples = int(w.sum())
                compute_times.append(max(round_compute))
                n_edge_done += 1
                # tier model for the next edge round = this round's tier aggregate
                tier_state = {**unflatten_to_state_dict(tier_flat, layout, dtypes=dtypes), **tier_bn}
                tier_model.load_state_dict(tier_state, strict=False)

            if n_edge_done == 0:
                continue    # empty tier this round (extreme non-IID skew)

            tier_results.append({
                "tier": tier, "tier_id": ti,
                "delta": tier_flat - global_flat,
                "bn": tier_bn,
                "samples": tier_samples,
                "num_clients": len(pool),
                "edge_rounds": n_edge_done,
                "intra_tier_bytes": int(msg_bytes * 2),         # uplink + downlink (paper eq. 8)
                "max_client_compute_time_sec": float(sum(compute_times)),
            })

        if not tier_results:
            continue

        # ---------- Cloud level: privacy stack over tier updates ----------
        active_ids = [r["tier_id"] for r in tier_results]
        total_samples = sum(r["samples"] for r in tier_results)
        payloads, tier_records = [], []
        encrypt_total, offset, true_len = 0.0, None, None

        for r in tier_results:
            update = r["delta"] * (r["samples"] / total_samples)

            norm = np.linalg.norm(update)                        # DP sensitivity clipping
            if norm > dp_clip:
                update = update * (dp_clip / norm)

            if dp_enabled:                                       # distributed Gaussian noise
                std = noise_mult * dp_clip / np.sqrt(len(tier_results))
                update = update + rng.normal(0.0, std, size=update.shape)

            plaintext_bytes = update.astype(np.float32).nbytes
            payload = update
            if sa_enabled:                                       # pairwise additive masking
                payload = mask_update(master_secret, round_idx, r["tier_id"], active_ids, payload, mask_scale)

            enc_time, sent_bytes = 0.0, plaintext_bytes
            if phe_enabled:
                q, offset = pu.quantize(payload, pcfg["quant_scale"], pcfg["quant_clip"])
                packed, true_len = pu.pack(q, pcfg["pack_batch_size"], pcfg["slot_bits"])
                print(f"[hfl-no-td] round {round_idx + 1}: tier {r['tier']} encrypting "
                      f"{len(packed)} ciphertexts...", flush=True)
                payload, enc_time = pu.encrypt_packed(packed, pub)
                sent_bytes = pu.ciphertext_bytes(payload)
                encrypt_total += enc_time
            payloads.append(payload)

            tier_records.append({
                "tier": r["tier"], "num_clients": r["num_clients"], "num_samples": r["samples"],
                "edge_rounds": r["edge_rounds"],
                "max_client_compute_time_sec": r["max_client_compute_time_sec"],
                "intra_tier_bytes": r["intra_tier_bytes"],
                "plaintext_message_bytes": int(plaintext_bytes),
                "transmitted_message_bytes": int(sent_bytes),
                "encrypt_time_sec": enc_time,
            })

        decrypt_time = 0.0
        if phe_enabled:
            summed = pu.sum_ciphertexts_across_clients(payloads)
            ints, decrypt_time = pu.decrypt_summed(summed, priv)
            unpacked = pu.unpack(ints, pcfg["pack_batch_size"], pcfg["slot_bits"], true_len)
            recovered = pu.dequantize_sum(unpacked, len(payloads), offset, pcfg["quant_scale"])
        else:
            recovered = np.sum(payloads, axis=0)   # SecAgg masks cancel exactly in this sum

        if args.server_opt == "fedadam":
            new_flat = fedadam.step(global_flat, recovered)
        else:                                      # plain weighted-delta step == FedAvg
            new_flat = global_flat + recovered

        new_bn = average_bn_buffers([r["bn"] for r in tier_results], [r["samples"] for r in tier_results])
        new_state = {**unflatten_to_state_dict(new_flat, layout, dtypes=dtypes), **new_bn}
        model.load_state_dict(new_state, strict=False)

        val = evaluate_on_subset(model, x_val_t, y_val_t, HEAD, NUM_GLOBAL_CLASSES, device)
        eps = rdp.step(round_idx, noise_mult) if dp_enabled else None

        logger.log({
            "round": round_idx, "seed": seed, "alpha": alpha,
            "secure_aggregation_enabled": sa_enabled, "dp_enabled": dp_enabled,
            "phe_enabled": phe_enabled, "server_opt": args.server_opt,
            "tiers": tier_records,
            "total_intra_tier_bytes": sum(t["intra_tier_bytes"] for t in tier_records),
            "total_tier_to_cloud_bytes": sum(t["transmitted_message_bytes"] for t in tier_records),
            "total_encrypt_time_sec": encrypt_total, "decrypt_time_sec": decrypt_time,
            "cumulative_epsilon": eps, "target_delta": target_delta if dp_enabled else None,
            "val_accuracy": val["accuracy"], "val_f1_macro": val["f1_macro"],
            "val_recall_macro": val["recall_macro"], "val_recall_weighted": val["recall_weighted"],
        })
        print(f"[hfl-no-td] seed={seed} round {round_idx + 1}/{num_rounds} "
              f"val_acc={val['accuracy']:.4f} val_f1_macro={val['f1_macro']:.4f} "
              f"eps={'n/a' if eps is None else f'{eps:.2f}'}", flush=True)

    total_time = time.perf_counter() - start
    logger.to_csv(os.path.join(args.log_dir, f"hfl_no_td_seed{seed}_rounds.csv"))
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"hfl_no_td_seed{seed}.pth"))

    test = evaluate_on_subset(model, x_test_t, y_test_t, HEAD, NUM_GLOBAL_CLASSES, device)
    print(f"[hfl-no-td] seed={seed} TEST acc={test['accuracy']:.4f} "
          f"f1_w={test['f1_weighted']:.4f} f1_m={test['f1_macro']:.4f} "
          f"rec_w={test['recall_weighted']:.4f} rec_m={test['recall_macro']:.4f} time={total_time:.1f}s")
    return test, {"train_time_sec": total_time, "paillier_key_gen_time_sec": key_gen_time,
                  "privacy": rdp.summary() if dp_enabled else None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha (default: config default_alpha)")
    p.add_argument("--rounds", type=int, default=None, help="default: config training.rounds.cloud")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: config seeds")
    p.add_argument("--device", default=None)
    p.add_argument("--plain", action="store_true", help="disable SecAgg + DP + PHE (hierarchy only)")
    p.add_argument("--no-secagg", action="store_true", help="disable only SecAgg masking")
    p.add_argument("--no-dp", action="store_true", help="disable only DP clipping noise + accounting")
    p.add_argument("--no-phe", action="store_true", help="disable only Paillier (much faster)")
    p.add_argument("--noise-multiplier", type=float, default=None,
                   help="override differential_privacy.noise_multiplier")
    p.add_argument("--edge-rounds", type=int, nargs=3, metavar=("A", "B", "C"), default=[1, 1, 1],
                   help="intra-tier FedAvg rounds per cloud round for tiers A B C. "
                        "1 1 1 == flat FedAvg; 2 2 1 with 15 cloud rounds gives A=30/B=30/C=15 "
                        "client-training rounds, i.e. >= CLW-HFL's 30/20/15 budget")
    p.add_argument("--server-opt", choices=["fedadam", "fedavg"], default="fedavg",
                   help="default fedavg: FedAdam was unstable on the 10-class model")
    p.add_argument("--out-name", default="hfl_no_td_results.json")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = args.device or config["runtime"]["device"]
    seeds = args.seeds or config["seeds"]
    alpha = config["partitioning"]["default_alpha"] if args.alpha is None else args.alpha
    num_rounds = args.rounds or config["training"]["rounds"]["cloud"]
    args.log_dir, args.ckpt_dir = config["paths"]["logs_dir"], config["paths"]["checkpoints_dir"]

    label_map = load_label_map(config["paths"]["label_map_path"])
    assert label_map == CANONICAL_LABEL_MAP, "label_map mismatch -- wrong dataset?"

    d = config["paths"]["processed_dir"]
    npz = {s: np.load(os.path.join(d, f"{s}.npz")) for s in ("train", "val", "test")}
    data = (
        torch.from_numpy(npz["train"]["X"]).float(), torch.from_numpy(npz["train"]["y"]).long(),
        torch.from_numpy(npz["val"]["X"]).float(), torch.from_numpy(npz["val"]["y"]).long(),
        torch.from_numpy(npz["test"]["X"]).float(), torch.from_numpy(npz["test"]["y"]).long(),
    )

    per_seed, extras = [], []
    for seed in seeds:
        print(f"\n[hfl-no-td] === seed {seed} ===")
        m, extra = run_one_seed(config, seed, alpha, num_rounds, device, data, args)
        per_seed.append(m)
        extras.append({"seed": seed, **extra})

    agg = aggregate_seeds(per_seed)
    out = {
        "script": "hfl_no_task_decomposition", "seeds": seeds, "alpha": alpha, "rounds": num_rounds,
        "edge_rounds": dict(zip(TIERS, args.edge_rounds)), "plain": args.plain, "no_secagg": args.no_secagg, "no_dp": args.no_dp,
        "no_phe": args.no_phe, "noise_multiplier": args.noise_multiplier,
        "server_opt": args.server_opt,
        "per_seed_results": [{k: v for k, v in m.items() if k != "confusion_matrix"} for m in per_seed],
        "per_seed_extras": extras, "aggregated": agg,
    }
    out_path = os.path.join(args.log_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== HFL WITHOUT TASK DECOMPOSITION (mean +/- std over seeds) ===")
    for k, name in [("accuracy", "Accuracy"), ("f1_weighted", "F1 (Weighted)"), ("f1_macro", "F1 (Macro)"),
                    ("recall_weighted", "Recall (Weighted)"), ("recall_macro", "Recall (Macro)")]:
        print(f"{name:<18}: {agg[k]['mean']:.2f} +/- {agg[k]['std']:.3f}   ({agg[k]['mean']:.4f})")
    print(f"[hfl-no-td] saved -> {out_path}")


if __name__ == "__main__":
    main()