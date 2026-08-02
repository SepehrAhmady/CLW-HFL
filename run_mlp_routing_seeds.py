"""
run_mlp_routing_seeds.py  --  Run MLP routing classifier for seeds 13, 42, 2024
and compute mean +/- std for Table I.

Usage:
    python run_mlp_routing_seeds.py

Requires:
    - checkpoints/layer_A_final_seed{N}.pth (or layer_A_final.pth for seed 42)
    - checkpoints/layer_B_final_seed{N}.pth
    - checkpoints/cloud_global_seed{N}.pth
    - data/processed/train.npz, test.npz

Output: logs/mlp_routing_multiseed_results.json
"""

import json
import os
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

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


def make_routing_labels(y_global):
    return np.array([LAYER_TO_IDX[LABEL_TO_LAYER[int(l)]] for l in y_global])


def get_ckpt_paths(seed):
    if seed == 42 and (CHECKPOINT_DIR / "layer_A_final.pth").exists():
        return {
            "A": (CHECKPOINT_DIR / "layer_A_final.pth",   "head_binary",   5),
            "B": (CHECKPOINT_DIR / "layer_B_final.pth",   "head_category", 5),
            "C": (CHECKPOINT_DIR / "cloud_global.pth",    "head_binary",   2),
        }
    return {
        "A": (CHECKPOINT_DIR / f"layer_A_final_seed{seed}.pth",  "head_binary",   5),
        "B": (CHECKPOINT_DIR / f"layer_B_final_seed{seed}.pth",  "head_category", 5),
        "C": (CHECKPOINT_DIR / f"cloud_global_seed{seed}.pth",   "head_binary",   2),
    }


def load_layer_models(seed):
    import torch
    from common.model import MultiHeadIDSMLP
    from common.label_utils import build_layer_remap

    device = torch.device("cpu")
    ckpt = get_ckpt_paths(seed)
    models, remaps = {}, {}
    for layer, (path, head_name, n_cls) in ckpt.items():
        kwargs = dict(input_dim=194, hidden_dims=(128, 64))
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
    return models, remaps


def evaluate_with_routing(X_test, y_test, routing_preds, models, remaps):
    import torch
    y_pred = np.full(len(y_test), -1, dtype=np.int64)
    for layer_idx, layer in IDX_TO_LAYER.items():
        mask = (routing_preds == layer_idx)
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
    y_pred[y_pred == -1] = 6
    return y_pred


def main():
    print("=" * 60)
    print("MLP Routing Classifier — Multi-Seed Evaluation")
    print("=" * 60)

    # Load data once
    train = np.load("data/processed/train.npz")
    test  = np.load("data/processed/test.npz")
    X_train = train["X"].astype(np.float32)
    y_train = train["y"].astype(np.int64)
    X_test  = test["X"].astype(np.float32)
    y_test  = test["y"].astype(np.int64)

    routing_train = make_routing_labels(y_train)
    routing_test  = make_routing_labels(y_test)

    per_seed_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Train MLP router with this seed
        print(f"  Training MLP router (random_state={seed})...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(X_train, routing_train)

        # Routing accuracy
        routing_pred = mlp.predict(X_test)
        route_acc = accuracy_score(routing_test, routing_pred)
        print(f"  Router accuracy: {route_acc:.4f}")

        # Load layer checkpoints for this seed
        print(f"  Loading layer checkpoints for seed {seed}...")
        models, remaps = load_layer_models(seed)

        # Evaluate end-to-end
        y_pred = evaluate_with_routing(X_test, y_test, routing_pred, models, remaps)

        acc     = float(accuracy_score(y_test, y_pred))
        f1_mac  = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
        f1_wei  = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        print(f"  Accuracy:     {acc:.4f}")
        print(f"  F1 (Macro):   {f1_mac:.4f}")
        print(f"  F1 (Weighted):{f1_wei:.4f}")

        per_seed_results.append({
            "seed": seed,
            "routing_accuracy": float(route_acc),
            "accuracy":    acc,
            "f1_macro":    f1_mac,
            "f1_weighted": f1_wei,
        })

    # Aggregate
    accs    = [r["accuracy"]    for r in per_seed_results]
    f1_macs = [r["f1_macro"]    for r in per_seed_results]
    f1_weis = [r["f1_weighted"] for r in per_seed_results]
    route_accs = [r["routing_accuracy"] for r in per_seed_results]

    aggregated = {
        "routing_accuracy": {"mean": float(np.mean(route_accs)), "std": float(np.std(route_accs))},
        "accuracy":         {"mean": float(np.mean(accs)),       "std": float(np.std(accs))},
        "f1_macro":         {"mean": float(np.mean(f1_macs)),    "std": float(np.std(f1_macs))},
        "f1_weighted":      {"mean": float(np.mean(f1_weis)),    "std": float(np.std(f1_weis))},
    }

    results = {
        "seeds": SEEDS,
        "per_seed_results": per_seed_results,
        "aggregated": aggregated,
    }

    out = LOG_DIR / "mlp_routing_multiseed_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS — MLP Routing (3 seeds)")
    print("=" * 60)
    print(f"Routing Accuracy: {aggregated['routing_accuracy']['mean']:.4f} ± {aggregated['routing_accuracy']['std']:.4f}")
    print(f"Accuracy:         {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
    print(f"F1 (Macro):       {aggregated['f1_macro']['mean']:.4f} ± {aggregated['f1_macro']['std']:.4f}")
    print(f"F1 (Weighted):    {aggregated['f1_weighted']['mean']:.4f} ± {aggregated['f1_weighted']['std']:.4f}")
    print(f"\nSaved: {out}")
    print("\nNext: send the output JSON to update Table I.")


if __name__ == "__main__":
    main()
