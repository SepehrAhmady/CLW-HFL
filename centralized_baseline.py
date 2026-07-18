"""centralized_baseline.py -- CLW-HFL Part 5, section 3.5.

Single model, full unrestricted access to ALL training data (all 10 classes),
trained end-to-end with no federation. Upper-bound reference for Table I.

Architecture: MultiHeadIDSMLP (same backbone) with head_fine used as the
10-class output head. Total epochs = training.rounds.cloud * training.local_epochs
(approx. same compute budget as CLW-HFL).

Runs at all 3 seeds from config; reports aggregate_seeds() mean ± std.

Output:
    logs/centralized_baseline_results.json

Example:
    python centralized_baseline.py
    python centralized_baseline.py --config config.yaml --device cpu
"""
import argparse
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
from common.logging_utils import RoundLogger


@torch.no_grad()
def evaluate(model, x_t, y_t, device, batch_size=2048):
    model.eval()
    preds = []
    for start in range(0, len(x_t), batch_size):
        xb = x_t[start:start + batch_size].to(device)
        preds.append(model(xb, head="head_fine").argmax(dim=-1).cpu().numpy())
    return compute_metrics(y_t.numpy(), np.concatenate(preds), labels=list(range(NUM_GLOBAL_CLASSES)))


def run_one_seed(config, seed, device, x_train_t, y_train_t, x_val_t, y_val_t,
                 x_test_t, y_test_t, log_dir, total_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = config["model"]["input_dim"]
    hidden_dims = tuple(config["model"]["hidden_dims"])
    batch_size = config["training"]["batch_size"]
    opt_cfg = config["training"]["optimizer"]
    grad_clip = config["training"]["grad_clip_norm"]

    model = MultiHeadIDSMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes_fine=NUM_GLOBAL_CLASSES,
    ).to(device)

    class_counts = np.bincount(y_train_t.numpy(), minlength=NUM_GLOBAL_CLASSES)
    loss_fn = ClassBalancedFocalLoss(
        class_counts,
        beta=config["training"]["focal_loss"]["beta"],
        gamma=config["training"]["focal_loss"]["gamma"],
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    # Cosine LR schedule: decays lr gradually over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    logger = RoundLogger(os.path.join(log_dir, f"centralized_seed{seed}.jsonl"))
    start_time = time.perf_counter()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(len(x_train_t))
        for start in range(0, len(x_train_t), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) <= 1:
                continue
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, head="head_fine"), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        # Evaluate on VAL set every epoch (not train set -- train accuracy is misleading)
        val_m = evaluate(model, x_val_t, y_val_t, device)
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, total_epochs // 5) == 0 or epoch == total_epochs - 1:
            print(f"  [seed={seed}] epoch {epoch + 1}/{total_epochs}  "
                  f"val_acc={val_m['accuracy']:.4f}  val_f1_macro={val_m['f1_macro']:.4f}  "
                  f"best_f1={best_val_f1:.4f}", flush=True)
        logger.log({"epoch": epoch, "seed": seed,
                    "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"]})

    # Load best checkpoint (by val F1) before final test evaluation
    model.load_state_dict(best_state)
    total_time = time.perf_counter() - start_time
    test_metrics = evaluate(model, x_test_t, y_test_t, device)
    print(f"  [seed={seed}] TEST acc={test_metrics['accuracy']:.4f}  "
          f"f1_macro={test_metrics['f1_macro']:.4f}  best_val_f1={best_val_f1:.4f}  time={total_time:.1f}s")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="Override config runtime.device")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs (default: rounds.cloud * local_epochs)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device or config["runtime"]["device"]
    seeds = config["seeds"]

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
    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)

    total_epochs = args.epochs or (config["training"]["rounds"]["cloud"] * config["training"]["local_epochs"])
    print(f"[centralized] {len(seeds)} seeds x {total_epochs} epochs, device={device}")
    print(f"[centralized] val set used for best-checkpoint selection (not train set)")

    per_seed_results = []
    for seed in seeds:
        print(f"\n[centralized] === seed {seed} ===")
        metrics = run_one_seed(config, seed, device,
                               x_train_t, y_train_t, x_val_t, y_val_t,
                               x_test_t, y_test_t, log_dir, total_epochs)
        per_seed_results.append(metrics)

    aggregated = aggregate_seeds(per_seed_results)

    output = {
        "script": "centralized_baseline",
        "seeds": seeds,
        "total_epochs": total_epochs,
        "per_seed_results": [
            {k: v for k, v in m.items() if k != "confusion_matrix"}
            for m in per_seed_results
        ],
        "aggregated": aggregated,
    }
    out_path = os.path.join(log_dir, "centralized_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== CENTRALIZED BASELINE (Table I) ===")
    print(f"Accuracy:      {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
    print(f"F1 (macro):    {aggregated['f1_macro']['mean']:.4f} ± {aggregated['f1_macro']['std']:.4f}")
    print(f"F1 (weighted): {aggregated['f1_weighted']['mean']:.4f} ± {aggregated['f1_weighted']['std']:.4f}")
    print(f"Recall (macro): {aggregated['recall_macro']['mean']:.4f} ± {aggregated['recall_macro']['std']:.4f}")
    print(f"\n[centralized] saved -> {out_path}")


if __name__ == "__main__":
    main()