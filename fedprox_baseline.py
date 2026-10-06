"""fedprox_baseline.py -- FedProx baseline (Li et al., MLSys 2020).

Identical protocol to flat_fedavg_baseline.py (same 50 clients, same Dirichlet
partition, same model / loss / optimizer / rounds / state_dict averaging /
evaluation) with ONE change: each client minimises

    local_loss(w) + (mu / 2) * || w - w_global ||^2

where w_global is the model received at the start of the round. With mu = 0
this reduces exactly to FedAvg (verified against flat_fedavg_baseline.py).

The proximal term is applied as its gradient, mu * (w - w_global), added to
the parameters' gradients BEFORE gradient clipping (so it is clipped together
with the task gradient, exactly as if it were part of the loss). It is applied
only to parameters that actually receive a task gradient, so the unused
classification heads of MultiHeadIDSMLP are left untouched, as in FedAvg.

mu tuning: pass several values, e.g. `--mu 0.001 0.01 0.1 1.0`. The script
reports every mu and marks the best one by mean FINAL VALIDATION macro-F1
(never by test). Note val is an optimistic random carve-out (see
check_leakage.py); the choice only ranks mu values, test is never used to pick.

Outputs:
    logs/fedprox_mu<mu>_seed<s>.jsonl      (per-round logs)
    logs/fedprox_results.json              (all mu + selected mu)

Example:
    python fedprox_baseline.py                          # mu = 0.01
    python fedprox_baseline.py --mu 0.001 0.01 0.1 1.0
    python fedprox_baseline.py --mu 0.0 --seeds 42 --rounds 3   # == FedAvg check
"""
import argparse
import copy
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
from common.fedavg import weighted_average_state_dicts
from common.logging_utils import RoundLogger, state_dict_size_bytes, timer
from flat_fedavg_baseline import evaluate


def local_train_prox(model, x_t, y_t, idxs, opt_cfg, batch_size, local_epochs,
                     grad_clip, loss_fn, device, mu):
    """Same as flat_fedavg_baseline.local_train plus the FedProx proximal term."""
    model.train()
    global_params = [p.detach().clone() for p in model.parameters()]   # w_global
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    for _ in range(local_epochs):
        perm = torch.randperm(len(idxs))
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[perm[start:start + batch_size]]
            if len(batch_idxs) <= 1:
                continue
            xb = x_t[batch_idxs].to(device)
            yb = y_t[batch_idxs].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, head="head_fine"), yb)
            loss.backward()
            if mu > 0.0:
                with torch.no_grad():
                    for p, g in zip(model.parameters(), global_params):
                        if p.grad is not None:
                            p.grad.add_(p - g, alpha=mu)       # d/dw [(mu/2)||w-g||^2]
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    return model


def run_one_seed(config, seed, mu, num_rounds, device, data, log_dir):
    x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, y_test_t = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    batch_size = config["training"]["batch_size"]
    local_epochs = config["training"]["local_epochs"]
    opt_cfg = config["training"]["optimizer"]
    grad_clip = config["training"]["grad_clip_norm"]
    alpha = config["partitioning"]["default_alpha"]
    client_fraction = config["clients"].get("client_fraction_per_round", 1.0)
    num_clients = (config["clients"]["layer_A"] + config["clients"]["layer_B"]
                   + config["clients"]["layer_C"])

    model = MultiHeadIDSMLP(
        input_dim=config["model"]["input_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        num_classes_fine=NUM_GLOBAL_CLASSES,
    ).to(device)

    y_np = y_train_t.numpy()
    loss_fn = ClassBalancedFocalLoss(
        np.bincount(y_np, minlength=NUM_GLOBAL_CLASSES),
        beta=config["training"]["focal_loss"]["beta"],
        gamma=config["training"]["focal_loss"]["gamma"],
    )
    client_indices = dirichlet_partition(y_np, num_clients=num_clients, alpha=alpha, seed=seed)

    logger = RoundLogger(os.path.join(log_dir, f"fedprox_mu{mu}_seed{seed}.jsonl"))
    start_time = time.perf_counter()

    for round_idx in range(num_rounds):
        n_selected = max(1, int(round(client_fraction * num_clients)))
        selected = rng.choice(num_clients, size=n_selected, replace=False)

        states, weights, times, msgs = [], [], [], []
        for cid in selected:
            idxs = client_indices[cid]
            if len(idxs) == 0:
                continue
            local_model = copy.deepcopy(model)
            with timer() as t:
                local_model = local_train_prox(
                    local_model, x_train_t, y_train_t, torch.from_numpy(idxs).long(),
                    opt_cfg, batch_size, local_epochs, grad_clip, loss_fn, device, mu,
                )
            states.append(local_model.state_dict())
            weights.append(len(idxs))
            times.append(t.elapsed)
            msgs.append(state_dict_size_bytes(local_model.state_dict()))
        if not states:
            continue

        model.load_state_dict(weighted_average_state_dicts(states, weights))
        val = evaluate(model, x_val_t, y_val_t, device)
        logger.log({
            "round": round_idx, "seed": seed, "mu": mu,
            "num_clients_this_round": len(states),
            "max_client_compute_time_sec": max(times),
            "total_message_bytes": sum(msgs) * 2,      # uplink + downlink (paper eq. 8)
            "val_accuracy": val["accuracy"], "val_f1_macro": val["f1_macro"],
        })
        if (round_idx + 1) % max(1, num_rounds // 5) == 0 or round_idx == num_rounds - 1:
            print(f"  [mu={mu} seed={seed}] round {round_idx + 1}/{num_rounds}  "
                  f"val_acc={val['accuracy']:.4f}  val_f1_macro={val['f1_macro']:.4f}", flush=True)

    total_time = time.perf_counter() - start_time
    val = evaluate(model, x_val_t, y_val_t, device)
    test = evaluate(model, x_test_t, y_test_t, device)
    print(f"  [mu={mu} seed={seed}] TEST acc={test['accuracy']:.4f} f1_w={test['f1_weighted']:.4f} "
          f"f1_m={test['f1_macro']:.4f} rec_w={test['recall_weighted']:.4f} "
          f"rec_m={test['recall_macro']:.4f}  time={total_time:.1f}s")
    return test, val


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--mu", type=float, nargs="+", default=[0.01],
                   help="proximal coefficient(s); several values are tuned on VAL macro-F1")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--rounds", type=int, default=None, help="default: training.rounds.cloud")
    p.add_argument("--out-name", default="fedprox_results.json")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = args.device or config["runtime"]["device"]
    seeds = args.seeds or config["seeds"]
    num_rounds = args.rounds or config["training"]["rounds"]["cloud"]
    log_dir = config["paths"]["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)

    assert load_label_map(config["paths"]["label_map_path"]) == CANONICAL_LABEL_MAP, \
        "label_map mismatch -- wrong dataset?"
    d = config["paths"]["processed_dir"]
    npz = {s: np.load(os.path.join(d, f"{s}.npz")) for s in ("train", "val", "test")}
    data = (
        torch.from_numpy(npz["train"]["X"]).float(), torch.from_numpy(npz["train"]["y"]).long(),
        torch.from_numpy(npz["val"]["X"]).float(), torch.from_numpy(npz["val"]["y"]).long(),
        torch.from_numpy(npz["test"]["X"]).float(), torch.from_numpy(npz["test"]["y"]).long(),
    )

    print(f"[fedprox] mu={args.mu} seeds={seeds} rounds={num_rounds} device={device}")
    results = {}
    for mu in args.mu:
        tests, vals = [], []
        for seed in seeds:
            print(f"\n[fedprox] === mu={mu} seed {seed} ===")
            t, v = run_one_seed(config, seed, mu, num_rounds, device, data, log_dir)
            tests.append(t)
            vals.append(v)
        results[mu] = {
            "aggregated_test": aggregate_seeds(tests),
            "mean_final_val_f1_macro": float(np.mean([v["f1_macro"] for v in vals])),
            "per_seed_test": [{k: x for k, x in m.items() if k != "confusion_matrix"} for m in tests],
        }

    best_mu = max(results, key=lambda m: results[m]["mean_final_val_f1_macro"])
    out = {
        "script": "fedprox_baseline", "seeds": seeds, "rounds": num_rounds,
        "mu_grid": args.mu, "selected_mu_by_val_f1_macro": best_mu,
        "per_mu": {str(m): r for m, r in results.items()},
        "aggregated": results[best_mu]["aggregated_test"],
    }
    out_path = os.path.join(log_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    keys = [("accuracy", "Acc"), ("f1_weighted", "F1w"), ("f1_macro", "F1m"),
            ("recall_weighted", "Rw"), ("recall_macro", "Rm")]
    print("\n=== FEDPROX (test, mean +/- std over seeds) ===")
    print(f"{'mu':>8} | " + " | ".join(f"{n:>13}" for _, n in keys) + " | val F1m")
    for mu, r in results.items():
        a = r["aggregated_test"]
        row = " | ".join(f"{a[k]['mean']:.3f}+/-{a[k]['std']:.3f}" for k, _ in keys)
        mark = "  <- selected (by val)" if mu == best_mu else ""
        print(f"{mu:>8g} | {row} | {r['mean_final_val_f1_macro']:.4f}{mark}")
    print(f"\n[fedprox] saved -> {out_path}")


if __name__ == "__main__":
    main()
