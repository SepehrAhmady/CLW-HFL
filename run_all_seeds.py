"""
run_all_seeds.py  --  Run CLW-HFL full pipeline for seeds 13, 42, 2024
and collect mean +/- std for Table I.

Usage:
    python run_all_seeds.py

Skips seed 42 if checkpoints already exist (re-uses existing results).
Output: logs/clwhfl_multiseed_results.json
"""

import json
import os
import time
import subprocess
import sys
import numpy as np
from pathlib import Path

SEEDS = [13, 42, 2024]
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
    "Analysis":0,"Backdoor":1,"DoS":2,"Exploits":3,"Fuzzers":4,
    "Generic":5,"Normal":6,"Reconnaissance":7,"Shellcode":8,"Worms":9
}
LAYER_ATTACK_MAP = {
    "A": ["Exploits","Shellcode","Fuzzers","Generic","Normal"],
    "B": ["Reconnaissance","DoS","Worms","Analysis","Normal"],
    "C": ["Backdoor","Normal"],
}
LABEL_TO_LAYER = {}
for layer, classes in LAYER_ATTACK_MAP.items():
    for cls in classes:
        gid = LABEL_MAP[cls]
        if gid not in LABEL_TO_LAYER:
            LABEL_TO_LAYER[gid] = layer

LAYER_TO_IDX = {"A":0,"B":1,"C":2}
IDX_TO_LAYER = {0:"A",1:"B",2:"C"}


def run_cmd(cmd, desc):
    print(f"\n[RUN] {desc}")
    print(f"      {' '.join(cmd)}")
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"[ERROR] Command failed (code {result.returncode})")
        sys.exit(1)
    print(f"[OK] Done in {elapsed/60:.1f} min")
    return elapsed


def checkpoints_exist(seed):
    """Check if all checkpoints for this seed already exist."""
    needed = [
        CHECKPOINT_DIR / f"layer_A_final_seed{seed}.pth",
        CHECKPOINT_DIR / f"layer_B_final_seed{seed}.pth",
        CHECKPOINT_DIR / f"layer_C_final_seed{seed}.pth",
        CHECKPOINT_DIR / f"cloud_global_seed{seed}.pth",
    ]
    # Also accept seed42 with original names (from previous run)
    if seed == 42:
        orig = [
            CHECKPOINT_DIR / "layer_A_final.pth",
            CHECKPOINT_DIR / "layer_B_final.pth",
            CHECKPOINT_DIR / "layer_C_final.pth",
            CHECKPOINT_DIR / "cloud_global.pth",
        ]
        if all(p.exists() for p in orig):
            return True
    return all(p.exists() for p in needed)


def evaluate_seed(seed):
    """Run evaluate_global for this seed and return metrics."""
    import torch
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    # Determine checkpoint paths
    if seed == 42 and (CHECKPOINT_DIR / "layer_A_final.pth").exists():
        ckpt = {
            "A": (CHECKPOINT_DIR / "layer_A_final.pth",   "head_binary",   5),
            "B": (CHECKPOINT_DIR / "layer_B_final.pth",   "head_category", 5),
            "C": (CHECKPOINT_DIR / "cloud_global.pth",    "head_binary",   2),
        }
    else:
        ckpt = {
            "A": (CHECKPOINT_DIR / f"layer_A_final_seed{seed}.pth",  "head_binary",   5),
            "B": (CHECKPOINT_DIR / f"layer_B_final_seed{seed}.pth",  "head_category", 5),
            "C": (CHECKPOINT_DIR / f"cloud_global_seed{seed}.pth",   "head_binary",   2),
        }

    from common.model import MultiHeadIDSMLP
    from common.label_utils import build_layer_remap

    device = torch.device("cpu")
    models = {}
    remaps = {}
    for layer, (path, head_name, n_cls) in ckpt.items():
        kwargs = dict(input_dim=194, hidden_dims=(128,64))
        if head_name == "head_binary":
            kwargs["num_classes_binary"] = n_cls
        else:
            kwargs["num_classes_category"] = n_cls
        model = MultiHeadIDSMLP(**kwargs).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        models[layer] = (model, head_name)
        g2l, l2g = build_layer_remap(layer, LABEL_MAP)
        remaps[layer] = (g2l, l2g)

    # Load test data
    test = np.load("data/processed/test.npz")
    X_test, y_test = test["X"].astype(np.float32), test["y"].astype(np.int64)

    # Oracle routing evaluation
    y_pred = np.full(len(y_test), -1, dtype=np.int64)
    for layer in ["A","B","C"]:
        owned_globals = set()
        for cls in LAYER_ATTACK_MAP[layer]:
            gid = LABEL_MAP[cls]
            if gid not in owned_globals:
                owned_globals.add(gid)

        mask = np.isin(y_test, list(owned_globals))
        # Remove samples already assigned
        mask = mask & (y_pred == -1)
        if mask.sum() == 0:
            continue

        model, head_name = models[layer]
        g2l, l2g = remaps[layer]
        X_sub = torch.from_numpy(X_test[mask]).float()

        with torch.no_grad():
            logits = model(X_sub, head=head_name)
            local_preds = logits.argmax(dim=1).numpy()

        global_preds = np.array([l2g.get(int(p), 6) for p in local_preds])
        y_pred[mask] = global_preds

    # Handle any unassigned (shouldn't happen)
    y_pred[y_pred == -1] = 6

    acc      = float(accuracy_score(y_test, y_pred))
    f1_mac   = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
    f1_wei   = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_mac  = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
    rec_wei  = float(accuracy_score(y_test, y_pred))

    return {"seed": seed, "accuracy": acc, "f1_macro": f1_mac,
            "f1_weighted": f1_wei, "recall_macro": rec_mac,
            "recall_weighted": rec_wei}


def main():
    print("=" * 60)
    print("CLW-HFL Multi-Seed Evaluation (seeds: 13, 42, 2024)")
    print("=" * 60)

    total_start = time.perf_counter()
    per_seed_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        if checkpoints_exist(seed):
            print(f"[SKIP] Checkpoints found for seed {seed} — skipping training")
        else:
            # Run training pipeline
            run_cmd(
                ["py", "layer_A_federated.py", "--seed", str(seed), "--alpha", "0.5"],
                f"Layer A training (seed={seed})"
            )
            # Rename checkpoint
            if seed != 42:
                os.rename(CHECKPOINT_DIR/"layer_A_final.pth",
                          CHECKPOINT_DIR/f"layer_A_final_seed{seed}.pth")

            run_cmd(
                ["py", "layer_B_federated.py", "--seed", str(seed), "--alpha", "0.5"],
                f"Layer B training (seed={seed})"
            )
            if seed != 42:
                os.rename(CHECKPOINT_DIR/"layer_B_final.pth",
                          CHECKPOINT_DIR/f"layer_B_final_seed{seed}.pth")

            run_cmd(
                ["py", "layer_C_federated.py", "--seed", str(seed), "--alpha", "0.5"],
                f"Layer C training (seed={seed})"
            )
            if seed != 42:
                os.rename(CHECKPOINT_DIR/"layer_C_final.pth",
                          CHECKPOINT_DIR/f"layer_C_final_seed{seed}.pth")

            run_cmd(
                ["py", "cloud_aggregate.py", "--seed", str(seed)],
                f"Cloud aggregation (seed={seed})"
            )
            if seed != 42:
                os.rename(CHECKPOINT_DIR/"cloud_global.pth",
                          CHECKPOINT_DIR/f"cloud_global_seed{seed}.pth")

        # Evaluate
        print(f"\n[EVAL] Evaluating seed {seed}...")
        metrics = evaluate_seed(seed)
        per_seed_results.append(metrics)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  F1 (Macro):  {metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")

    # Aggregate
    accs     = [r["accuracy"]     for r in per_seed_results]
    f1_macs  = [r["f1_macro"]     for r in per_seed_results]
    f1_weis  = [r["f1_weighted"]  for r in per_seed_results]

    aggregated = {
        "accuracy":     {"mean": float(np.mean(accs)),    "std": float(np.std(accs))},
        "f1_macro":     {"mean": float(np.mean(f1_macs)), "std": float(np.std(f1_macs))},
        "f1_weighted":  {"mean": float(np.mean(f1_weis)), "std": float(np.std(f1_weis))},
    }

    results = {
        "seeds": SEEDS,
        "per_seed_results": per_seed_results,
        "aggregated": aggregated,
    }

    out_path = LOG_DIR / "clwhfl_multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total_min = (time.perf_counter() - total_start) / 60

    print("\n" + "="*60)
    print("FINAL RESULTS — CLW-HFL (oracle routing, 3 seeds)")
    print("="*60)
    print(f"Accuracy:     {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
    print(f"F1 (Macro):   {aggregated['f1_macro']['mean']:.4f} ± {aggregated['f1_macro']['std']:.4f}")
    print(f"F1 (Weighted):{aggregated['f1_weighted']['mean']:.4f} ± {aggregated['f1_weighted']['std']:.4f}")
    print(f"\nTotal time: {total_min:.1f} min")
    print(f"Saved: {out_path}")
    print("\nNext: paste these numbers into Table I of the paper.")


if __name__ == "__main__":
    main()
