"""flat_fedavg_baseline.py -- CLW-HFL Part 5, section 3.5.

Standard two-tier FedAvg: all 50 clients (30+10+10) flattened into one tier,
training a single 10-class MultiHeadIDSMLP (head_fine) with plain weighted
FedAvg and gradient clipping only -- no hierarchy, no secure aggregation,
no DP, no PHE. Middle row of Table I.

Runs at all 3 seeds from config; reports aggregate_seeds() mean ± std.
Per-round logging includes val metrics and communication-cost stats
(message size × clients × 2 = uplink + downlink, per paper eq. 8).

Output:
    logs/flat_fedavg_results.json

Example:
    python flat_fedavg_baseline.py
    python flat_fedavg_baseline.py --config config.yaml --device cpu
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
from common.metrics import compute_metrics, aggregate_seeds
from common.partition import dirichlet_partition
from common.fedavg import weighted_average_state_dicts
from common.logging_utils import RoundLogger, state_dict_size_bytes, timer


@torch.no_grad()
def evaluate(model, x_t, y_t, device, batch_size=2048):
    model.eval()
    preds = []
    for start in range(0, len(x_t), batch_size):
        xb = x_t[start:start + batch_size].to(device)
        preds.append(model(xb, head="head_fine").argmax(dim=-1).cpu().numpy())
    return compute_metrics(y_t.numpy(), np.concatenate(preds), labels=list(range(NUM_GLOBAL_CLASSES)))


def local_train(model, x_t, y_t, idxs, opt_cfg, batch_size, local_epochs,
                grad_clip, loss_fn, device):
    model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    return model


def run_one_seed(config, seed, device, x_train_t, y_train_t, x_val_t, y_val_t,
                 x_test_t, y_test_t, log_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    input_dim = config["model"]["input_dim"]
    hidden_dims = tuple(config["model"]["hidden_dims"])
    batch_size = config["training"]["batch_size"]
    local_epochs = config["training"]["local_epochs"]
    num_rounds = config["training"]["rounds"]["cloud"]
    opt_cfg = config["training"]["optimizer"]
    grad_clip = config["training"]["grad_clip_norm"]
    alpha = config["partitioning"]["default_alpha"]
    client_fraction = config["clients"].get("client_fraction_per_round", 1.0)
    num_clients = (config["clients"]["layer_A"] +
                   config["clients"]["layer_B"] +
                   config["clients"]["layer_C"])   # 50 total

    model = MultiHeadIDSMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes_fine=NUM_GLOBAL_CLASSES,
    ).to(device)

    y_train_np = y_train_t.numpy()
    class_counts = np.bincount(y_train_np, minlength=NUM_GLOBAL_CLASSES)
    loss_fn = ClassBalancedFocalLoss(
        class_counts,
        beta=config["training"]["focal_loss"]["beta"],
        gamma=config["training"]["focal_loss"]["gamma"],
    )

    # Partition ALL training data (all 10 classes) across 50 clients
    client_indices = dirichlet_partition(y_train_np, num_clients=num_clients,
                                          alpha=alpha, seed=seed)

    logger = RoundLogger(os.path.join(log_dir, f"flat_fedavg_seed{seed}.jsonl"))
    start_time = time.perf_counter()

    for round_idx in range(num_rounds):
        n_selected = max(1, int(round(client_fraction * num_clients)))
        selected = rng.choice(num_clients, size=n_selected, replace=False)

        client_state_dicts, client_weights = [], []
        client_times, client_msg_bytes = [], []

        for cid in selected:
            idxs = client_indices[cid]
            if len(idxs) == 0:
                continue
            local_model = copy.deepcopy(model)
            with timer() as t:
                local_model = local_train(
                    local_model, x_train_t, y_train_t, torch.from_numpy(idxs).long(),
                    opt_cfg, batch_size, local_epochs, grad_clip, loss_fn, device,
                )
            # Communication cost: uplink (client->server) state_dict bytes
            # Total = bytes * n_clients * 2 (uplink + downlink, paper eq. 8)
            msg = state_dict_size_bytes(local_model.state_dict())
            client_state_dicts.append(local_model.state_dict())
            client_weights.append(len(idxs))
            client_times.append(t.elapsed)
            client_msg_bytes.append(msg)

        if not client_state_dicts:
            continue

        aggregated = weighted_average_state_dicts(client_state_dicts, client_weights)
        model.load_state_dict(aggregated)

        val_metrics = evaluate(model, x_val_t, y_val_t, device)
        # Paper eq. 8: comm_cost = msg_size * n_clients * 2 (both directions)
        total_comm = sum(client_msg_bytes) * 2

        record = {
            "round": round_idx,
            "seed": seed,
            "num_clients_this_round": len(client_state_dicts),
            "max_client_compute_time_sec": max(client_times),
            "total_message_bytes": total_comm,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        logger.log(record)

        if (round_idx + 1) % max(1, num_rounds // 5) == 0 or round_idx == num_rounds - 1:
            print(f"  [seed={seed}] round {round_idx + 1}/{num_rounds}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  val_f1_macro={val_metrics['f1_macro']:.4f}",
                  flush=True)

    total_time = time.perf_counter() - start_time
    test_metrics = evaluate(model, x_test_t, y_test_t, device)
    print(f"  [seed={seed}] TEST acc={test_metrics['accuracy']:.4f}  "
          f"f1_macro={test_metrics['f1_macro']:.4f}  time={total_time:.1f}s")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="Override config runtime.device")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config["runtime"]["device"]
    seeds = config["seeds"]

    # Dataset-identity guard (Part 1, section 1.1)
    label_map = load_label_map(config["paths"]["label_map_path"])
    assert label_map == CANONICAL_LABEL_MAP, "label_map mismatch -- wrong dataset?"

    processed_dir = config["paths"]["processed_dir"]
    train_npz = np.load(os.path.join(processed_dir, "train.npz"))
    val_npz = np.load(os.path.join(processed_dir, "val.npz"))
    test_npz = np.load(os.path.join(processed_dir, "test.npz"))
    x_train_t = torch.from_numpy(train_npz["X"]).float()
    y_train_t = torch.from_numpy(train_npz["y"]).long()
    x_val_t = torch.from_numpy(val_npz["X"]).float()
    y_val_t = torch.from_numpy(val_npz["y"]).long()
    x_test_t = torch.from_numpy(test_npz["X"]).float()
    y_test_t = torch.from_numpy(test_npz["y"]).long()

    log_dir = config["paths"]["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)

    num_clients = (config["clients"]["layer_A"] +
                   config["clients"]["layer_B"] +
                   config["clients"]["layer_C"])
    num_rounds = config["training"]["rounds"]["cloud"]
    print(f"[flat_fedavg] {len(seeds)} seeds x {num_rounds} rounds x {num_clients} clients, device={device}")

    per_seed_results = []
    for seed in seeds:
        print(f"\n[flat_fedavg] === seed {seed} ===")
        metrics = run_one_seed(config, seed, device,
                               x_train_t, y_train_t, x_val_t, y_val_t,
                               x_test_t, y_test_t, log_dir)
        per_seed_results.append(metrics)

    aggregated = aggregate_seeds(per_seed_results)

    output = {
        "script": "flat_fedavg_baseline",
        "seeds": seeds,
        "per_seed_results": [
            {k: v for k, v in m.items() if k != "confusion_matrix"}
            for m in per_seed_results
        ],
        "aggregated": aggregated,
    }
    out_path = os.path.join(log_dir, "flat_fedavg_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== FLAT FEDAVG BASELINE (Table I) ===")
    print(f"Accuracy:      {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
    print(f"F1 (macro):    {aggregated['f1_macro']['mean']:.4f} ± {aggregated['f1_macro']['std']:.4f}")
    print(f"F1 (weighted): {aggregated['f1_weighted']['mean']:.4f} ± {aggregated['f1_weighted']['std']:.4f}")
    print(f"Recall (macro): {aggregated['recall_macro']['mean']:.4f} ± {aggregated['recall_macro']['std']:.4f}")
    print(f"\n[flat_fedavg] saved -> {out_path}")


if __name__ == "__main__":
    main()
