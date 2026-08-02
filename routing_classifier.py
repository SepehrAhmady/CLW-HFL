"""
routing_classifier.py  --  CLW-HFL Learned Routing Classifier

Trains a lightweight routing classifier that, given a raw feature vector,
predicts which CLW-HFL layer (A, B, or C) should handle it -- WITHOUT
access to the ground-truth attack label. This replaces oracle routing and
produces a real (non-upper-bound) accuracy figure for Table I.

Usage:
    python routing_classifier.py

Outputs:
    logs/routing_classifier_results.json   (accuracy + F1 for each method)
    logs/routing_confusion.json            (confusion matrix)

The script evaluates CLW-HFL under three routing strategies:
    1. Oracle routing    (upper bound, uses true label)
    2. RF routing        (Random Forest, trained on train split)
    3. MLP routing       (small neural net, alternative)

All three are reported so the paper can compare honestly.
"""

import json
import os
import numpy as np
from pathlib import Path

# ── label map and layer assignment (must match label_utils.py) ────────────
LABEL_MAP = {
    "Analysis": 0, "Backdoor": 1, "DoS": 2, "Exploits": 3, "Fuzzers": 4,
    "Generic": 5, "Normal": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9
}

# Which global label belongs to which layer
LAYER_ATTACK_MAP = {
    "A": ["Exploits", "Shellcode", "Fuzzers", "Generic", "Normal"],
    "B": ["Reconnaissance", "DoS", "Worms", "Analysis", "Normal"],
    "C": ["Backdoor", "Normal"],
}

# Build global_label -> layer mapping
# Normal (6) is in all three; assign to A (largest layer, most Normal samples)
LABEL_TO_LAYER = {}
for layer, classes in LAYER_ATTACK_MAP.items():
    for cls in classes:
        gid = LABEL_MAP[cls]
        if gid not in LABEL_TO_LAYER:
            LABEL_TO_LAYER[gid] = layer

LAYER_TO_IDX = {"A": 0, "B": 1, "C": 2}
IDX_TO_LAYER = {0: "A", 1: "B", 2: "C"}

# Per-layer: which global labels it owns (for evaluation)
LAYER_OWNED = {}
seen = set()
for layer, classes in LAYER_ATTACK_MAP.items():
    owned = []
    for cls in classes:
        gid = LABEL_MAP[cls]
        if gid not in seen:
            owned.append(gid)
            seen.add(gid)
    LAYER_OWNED[layer] = owned


def load_data(processed_dir="data/processed"):
    """Load preprocessed train/test splits."""
    train = np.load(os.path.join(processed_dir, "train.npz"))
    test  = np.load(os.path.join(processed_dir, "test.npz"))
    return (train["X"].astype(np.float32), train["y"].astype(np.int64),
            test["X"].astype(np.float32),  test["y"].astype(np.int64))


def make_routing_labels(y_global):
    """Convert global class labels to routing labels (0=A, 1=B, 2=C)."""
    routing = np.array([LAYER_TO_IDX[LABEL_TO_LAYER[int(l)]] for l in y_global])
    return routing


def load_layer_checkpoints(checkpoint_dir="checkpoints"):
    """Load the three layer checkpoints and build per-layer predictors."""
    import torch
    from common.model import MultiHeadIDSMLP
    from common.label_utils import build_layer_remap, filter_and_remap

    device = torch.device("cpu")
    models = {}
    remaps = {}

    ckpt_map = {
        "A": ("layer_A_final.pth",   "head_binary",   5),
        "B": ("layer_B_final.pth",   "head_category", 5),
        "C": ("cloud_global.pth",    "head_binary",   2),
    }

    for layer, (fname, head_name, n_cls) in ckpt_map.items():
        path = os.path.join(checkpoint_dir, fname)
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


@np.vectorize
def route_oracle(label):
    return LAYER_TO_IDX[LABEL_TO_LAYER[int(label)]]


def evaluate_with_routing(X_test, y_test, routing_preds, models, remaps, device="cpu"):
    """
    Given routing predictions (0/1/2 per sample), run inference with the
    appropriate layer model and compute global accuracy + per-class F1.
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    y_pred_global = np.full(len(y_test), -1, dtype=np.int64)

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

        # Remap local predictions back to global label space
        global_preds = np.array([l2g.get(int(p), 6) for p in local_preds])
        y_pred_global[mask] = global_preds

    # Samples routed to wrong layer may predict a class not in that layer
    # Count routing errors as misclassifications (already handled by l2g fallback)
    acc = accuracy_score(y_test, y_pred_global)
    f1_macro    = f1_score(y_test, y_pred_global, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_global, average="weighted", zero_division=0)

    report = classification_report(
        y_test, y_pred_global,
        labels=list(range(10)),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0,
        output_dict=True
    )

    return {
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "per_class":    {k: v for k, v in report.items() if k in LABEL_MAP},
    }


def train_rf_router(X_train, y_train):
    """Train a Random Forest routing classifier."""
    from sklearn.ensemble import RandomForestClassifier
    routing_train = make_routing_labels(y_train)
    print(f"[router] Training Random Forest on {len(X_train)} samples...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, routing_train)
    train_acc = rf.score(X_train, routing_train)
    print(f"[router] RF train routing accuracy: {train_acc:.4f}")
    return rf


def train_mlp_router(X_train, y_train):
    """Train a lightweight MLP routing classifier."""
    from sklearn.neural_network import MLPClassifier
    routing_train = make_routing_labels(y_train)
    print(f"[router] Training MLP router on {len(X_train)} samples...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train, routing_train)
    train_acc = mlp.score(X_train, routing_train)
    print(f"[router] MLP train routing accuracy: {train_acc:.4f}")
    return mlp


def routing_accuracy(router, X_test, y_test):
    """How accurately does the router predict the correct layer?"""
    routing_true = make_routing_labels(y_test)
    routing_pred = router.predict(X_test)
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(routing_true, routing_pred))


def main():
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("CLW-HFL Routing Classifier Evaluation")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Load layer checkpoints
    print("\n[2] Loading layer checkpoints...")
    try:
        models, remaps = load_layer_checkpoints()
        print("    Loaded: layer_A_final.pth, layer_B_final.pth, cloud_global.pth")
    except Exception as e:
        print(f"    ERROR loading checkpoints: {e}")
        print("    Make sure checkpoints/ folder has the .pth files.")
        return

    results = {}

    # 3. Oracle routing (upper bound)
    print("\n[3] Oracle routing (upper bound)...")
    oracle_routing = route_oracle(y_test)
    oracle_metrics = evaluate_with_routing(X_test, y_test, oracle_routing, models, remaps)
    results["oracle_routing"] = oracle_metrics
    print(f"    Accuracy: {oracle_metrics['accuracy']:.4f} | "
          f"F1-Macro: {oracle_metrics['f1_macro']:.4f} | "
          f"F1-Weighted: {oracle_metrics['f1_weighted']:.4f}")

    # 4. RF router
    print("\n[4] Random Forest routing classifier...")
    rf = train_rf_router(X_train, y_train)
    rf_route_acc = routing_accuracy(rf, X_test, y_test)
    print(f"    Router accuracy on test set: {rf_route_acc:.4f}")
    rf_routing = rf.predict(X_test)
    rf_metrics = evaluate_with_routing(X_test, y_test, rf_routing, models, remaps)
    rf_metrics["routing_accuracy"] = rf_route_acc
    results["rf_routing"] = rf_metrics
    print(f"    Accuracy: {rf_metrics['accuracy']:.4f} | "
          f"F1-Macro: {rf_metrics['f1_macro']:.4f} | "
          f"F1-Weighted: {rf_metrics['f1_weighted']:.4f}")

    # 5. MLP router
    print("\n[5] MLP routing classifier...")
    mlp = train_mlp_router(X_train, y_train)
    mlp_route_acc = routing_accuracy(mlp, X_test, y_test)
    print(f"    Router accuracy on test set: {mlp_route_acc:.4f}")
    mlp_routing = mlp.predict(X_test)
    mlp_metrics = evaluate_with_routing(X_test, y_test, mlp_routing, models, remaps)
    mlp_metrics["routing_accuracy"] = mlp_route_acc
    results["mlp_routing"] = mlp_metrics
    print(f"    Accuracy: {mlp_metrics['accuracy']:.4f} | "
          f"F1-Macro: {mlp_metrics['f1_macro']:.4f} | "
          f"F1-Weighted: {mlp_metrics['f1_weighted']:.4f}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (for Table I update)")
    print("=" * 60)
    print(f"{'Configuration':<35} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}")
    print("-" * 70)
    print(f"{'CLW-HFL (oracle routing — upper bound)':<35} "
          f"{oracle_metrics['accuracy']:>10.4f} "
          f"{oracle_metrics['f1_macro']:>10.4f} "
          f"{oracle_metrics['f1_weighted']:>12.4f}")
    print(f"{'CLW-HFL (RF routing classifier)':<35} "
          f"{rf_metrics['accuracy']:>10.4f} "
          f"{rf_metrics['f1_macro']:>10.4f} "
          f"{rf_metrics['f1_weighted']:>12.4f}")
    print(f"{'CLW-HFL (MLP routing classifier)':<35} "
          f"{mlp_metrics['accuracy']:>10.4f} "
          f"{mlp_metrics['f1_macro']:>10.4f} "
          f"{mlp_metrics['f1_weighted']:>12.4f}")
    print(f"{'Centralized baseline':<35} {'0.6980':>10} {'0.4198':>10} {'0.7394':>12}")
    print(f"{'Flat FedAvg baseline':<35} {'0.6481':>10} {'0.3274':>10} {'0.6945':>12}")

    # 7. Save
    with open("logs/routing_classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[saved] logs/routing_classifier_results.json")
    print("\nNext step: paste the RF/MLP accuracy into Table I of the paper,")
    print("and add a sentence in III-C explaining learned routing replaces oracle.")


if __name__ == "__main__":
    main()