"""evaluate_global.py -- CLW-HFL Part 6 (the most critical fix in the spec).

Produces the unified global evaluation metrics for Table I, Figure 3, and
Figure 4. Because no single head was ever trained to distinguish all 10
classes, there is no "one model" to score on the full test set. Instead,
each test sample is ROUTED by its TRUE LABEL to the layer responsible for
that class (via LAYER_ATTACK_MAP), scored with that layer's own checkpoint,
and the prediction is remapped back to the canonical 10-class global label
space before being dropped into ONE accumulated confusion matrix.

Routing summary (Part 1, section 1.3 / Part 6, step 2):
  Layer A classes → layer_A_final.pth, head_binary   (5-way)
  Layer B classes → layer_B_final.pth, head_category (5-way)
  Layer C classes → cloud_global.pth,  head_binary   (2-way)
    ^--- IMPORTANT: Layer C uses cloud_global.pth (the post-Cloud-aggregation
         artifact), NOT layer_C_final.pth (plain FedAvg only). This is
         exactly the bug described in Part 2, item 1 that this rebuild
         exists to prevent.

METHODOLOGY NOTE (documented, not hidden): routing uses the TRUE label,
making this an oracle/upper-bound evaluation. It measures "best-case
accuracy given that a downstream router knows which layer to send each
packet to." In a real deployment a learned or rule-based classifier would
do the routing without access to the true label. A one-line caveat to this
effect is included in the generated metadata, and the README flags it.

Outputs:
    logs/evaluate_global_results.json   -- full metrics (Table I numbers)
    logs/evaluate_global_confusion.json -- 10x10 confusion matrix (Figure 3)
    logs/evaluate_per_layer.json        -- per-layer isolated metrics (Figure 5)

Example:
    python evaluate_global.py
    python evaluate_global.py --split val   # quick check during development
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.label_utils import (
    CANONICAL_LABEL_MAP,
    LAYER_ATTACK_MAP,
    LAYER_HEAD_NAME,
    build_layer_remap,
    layer_num_classes,
    load_label_map,
)
from common.model import build_model_for_layer
from common.metrics import compute_metrics


def load_layer_model(layer, checkpoint_path, config, device):
    """Load a layer's own final checkpoint into a fresh model of the correct
    shape. Always loads the full state_dict (strict=False is intentional and
    documented: the checkpoint only contains backbone + this layer's active
    head; the other two unused heads are left at their random-init values
    and are never called during evaluation, so this is harmless).
    """
    model = build_model_for_layer(
        layer,
        input_dim=config["model"]["input_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
    ).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'. "
            f"Run the appropriate training script first:\n"
            f"  Layer A -> python layer_A_federated.py\n"
            f"  Layer B -> python layer_B_federated.py\n"
            f"  Layer C -> python layer_C_federated.py\n"
            f"  Cloud   -> python cloud_aggregate.py\n"
            f"(Layer C's contribution to global eval uses cloud_global.pth, "
            f"NOT layer_C_final.pth -- see Part 2, item 1.)"
        )
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_layer_subset(model, x_np, head_name, device, batch_size=2048):
    """Run inference for a layer on its own subset of samples.
    Returns np.ndarray of local (layer-internal) predicted class indices.
    """
    model.eval()
    preds = []
    x_t = torch.from_numpy(x_np).float()
    for start in range(0, len(x_t), batch_size):
        xb = x_t[start:start + batch_size].to(device)
        logits = model(xb, head=head_name)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_global(config, split="test", device="cpu"):
    """
    Main evaluation routine. Returns (global_metrics, per_layer_metrics, meta).
    """
    processed_dir = config["paths"]["processed_dir"]
    checkpoint_dir = config["paths"]["checkpoints_dir"]
    label_map = load_label_map(config["paths"]["label_map_path"])

    npz = np.load(os.path.join(processed_dir, f"{split}.npz"))
    x_all = npz["X"]
    y_all = npz["y"]

    # --- Map from checkpoint file to each layer (Part 6, step 2) ---
    # Layer C deliberately uses cloud_global.pth, not layer_C_final.pth.
    checkpoint_map = {
        "A": os.path.join(checkpoint_dir, "layer_A_final.pth"),
        "B": os.path.join(checkpoint_dir, "layer_B_final.pth"),
        "C": os.path.join(checkpoint_dir, "cloud_global.pth"),   # NOT layer_C_final.pth
    }

    # --- Load all three checkpoints ---
    models = {}
    for layer, ckpt_path in checkpoint_map.items():
        print(f"[eval_global] loading {layer} <- {ckpt_path}")
        models[layer] = load_layer_model(layer, ckpt_path, config, device)

    # --- Build a global_to_layer lookup: for each canonical global label id,
    # which layer owns it? ---
    global_id_to_layer = {}
    for layer, classes in LAYER_ATTACK_MAP.items():
        for cls_name in classes:
            global_id = label_map[cls_name]
            # "Normal" appears in all three layers' LAYER_ATTACK_MAP, but for
            # routing purposes we need to pick exactly one layer per sample.
            # Per the spec, the routing key is the TRUE label (attack_cat).
            # Normal samples are handled by whichever layer's local task they
            # appear in -- but since Normal is in all three, we need a tie-
            # breaking rule. The spec doesn't specify one for Normal's routing,
            # so we use Layer A (the largest layer, most representative) for
            # Normal samples in the global eval. This is documented below.
            if cls_name == "Normal":
                if global_id not in global_id_to_layer:
                    global_id_to_layer[global_id] = layer  # first layer = A (dict order)
            else:
                global_id_to_layer[global_id] = layer

    # Verify every global label id is covered.
    assert set(global_id_to_layer.keys()) == set(label_map.values()), \
        f"Not all global label ids are routed: {set(label_map.values()) - set(global_id_to_layer.keys())}"

    # --- Route, predict, remap, accumulate ---
    # Global arrays for the unified confusion matrix (Part 6, step 4).
    all_y_true = []
    all_y_pred = []

    # Per-layer arrays for isolated metrics (Part 6, step 5 / Figure 5).
    per_layer_true = {"A": [], "B": [], "C": []}
    per_layer_pred = {"A": [], "B": [], "C": []}
    per_layer_local_true = {"A": [], "B": [], "C": []}  # local (layer-internal) indices
    per_layer_local_pred = {"A": [], "B": [], "C": []}

    for layer in ["A", "B", "C"]:
        global_to_local, local_to_global = build_layer_remap(layer, label_map)
        head_name = LAYER_HEAD_NAME[layer]
        num_classes = layer_num_classes(layer)

        # Select rows routed to this layer by true label.
        owned_global_ids = [gid for gid, lyr in global_id_to_layer.items() if lyr == layer]
        mask = np.isin(y_all, owned_global_ids)
        x_layer = x_all[mask]
        y_global_layer = y_all[mask]

        if len(x_layer) == 0:
            print(f"[eval_global] Layer {layer}: 0 samples routed -- skipping")
            continue

        # Local label indices for this layer's head.
        lut = np.full(int(y_global_layer.max()) + 1, fill_value=-1, dtype=np.int64)
        for g, l in global_to_local.items():
            if g <= int(y_global_layer.max()):
                lut[g] = l
        y_local_layer = lut[y_global_layer]
        assert (y_local_layer >= 0).all(), \
            f"Layer {layer}: some routed samples have unmapped global labels"

        print(f"[eval_global] Layer {layer}: {len(x_layer)} samples, "
              f"classes {LAYER_ATTACK_MAP[layer]}")

        # Predict with this layer's model and head.
        y_local_pred = predict_layer_subset(models[layer], x_layer, head_name, device)

        # Remap local predictions back to global label space (Part 6, step 3).
        y_global_pred = np.array([local_to_global[int(p)] for p in y_local_pred])

        # Accumulate into global arrays.
        all_y_true.extend(y_global_layer.tolist())
        all_y_pred.extend(y_global_pred.tolist())

        # Per-layer arrays (for Figure 5 isolated metrics).
        per_layer_true[layer] = y_global_layer.tolist()
        per_layer_pred[layer] = y_global_pred.tolist()
        per_layer_local_true[layer] = y_local_layer.tolist()
        per_layer_local_pred[layer] = y_local_pred.tolist()

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    assert len(all_y_true) == len(y_all), \
        f"Not all test samples were evaluated: {len(all_y_true)} vs {len(y_all)}"

    # --- Global metrics (Part 6, step 4) -- these are Table I numbers ---
    all_labels = list(range(len(label_map)))
    global_metrics = compute_metrics(all_y_true, all_y_pred, labels=all_labels)

    # --- Per-layer isolated metrics (Part 6, step 5) -- Figure 5 ---
    per_layer_metrics = {}
    for layer in ["A", "B", "C"]:
        if not per_layer_local_true[layer]:
            continue
        layer_labels = list(range(layer_num_classes(layer)))
        per_layer_metrics[layer] = compute_metrics(
            per_layer_local_true[layer],
            per_layer_local_pred[layer],
            labels=layer_labels,
        )

    # Reverse label map for human-readable class names in output
    id_to_name = {v: k for k, v in label_map.items()}

    meta = {
        "split": split,
        "n_samples_total": int(len(all_y_true)),
        "routing": {
            id_to_name[gid]: lyr
            for gid, lyr in global_id_to_layer.items()
        },
        "methodology_note": (
            "Oracle routing: each test sample is routed to its responsible "
            "layer using the TRUE label (not a predicted one). This is the "
            "upper-bound evaluation described in Part 6 of the CLW-HFL spec. "
            "In a real deployment, a separate routing classifier would be "
            "needed that does not have access to the true label."
        ),
        "checkpoint_used": {
            "A": checkpoint_map["A"],
            "B": checkpoint_map["B"],
            "C": checkpoint_map["C"] + " (cloud_global.pth -- post-Cloud-aggregation, NOT layer_C_final.pth)",
        },
        "global_metrics_summary": {
            "accuracy": global_metrics["accuracy"],
            "f1_macro": global_metrics["f1_macro"],
            "f1_weighted": global_metrics["f1_weighted"],
        },
    }

    return global_metrics, per_layer_metrics, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                         help="Which data split to evaluate on. Use 'val' for "
                              "quick development checks; 'test' for paper numbers.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config["runtime"]["device"]
    log_dir = config["paths"]["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)

    print(f"[eval_global] evaluating on split='{args.split}', device='{device}'")
    global_metrics, per_layer_metrics, meta = evaluate_global(config, split=args.split, device=device)

    # --- Print summary ---
    print("\n" + "="*60)
    print("GLOBAL EVALUATION RESULTS (Table I / Figure 3-4)")
    print("="*60)
    print(f"  Accuracy:          {global_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {global_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {global_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {global_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {global_metrics['f1_weighted']:.4f}")
    print()
    print("Per-class F1 (global label space):")
    id_to_name = {v: k for k, v in CANONICAL_LABEL_MAP.items()}
    for lid in sorted(global_metrics["per_class"].keys(), key=int):
        cls = id_to_name.get(int(lid), str(lid))
        f1 = global_metrics["per_class"][lid]["f1"]
        sup = global_metrics["per_class"][lid]["support"]
        routed_to = meta["routing"].get(cls, "?")
        print(f"  {cls:20s} (layer {routed_to}): F1={f1:.4f}  support={sup}")
    print()
    print("Per-layer isolated metrics (Figure 5):")
    for layer in ["A", "B", "C"]:
        if layer in per_layer_metrics:
            m = per_layer_metrics[layer]
            print(f"  Layer {layer}: acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")
    print("="*60)

    # --- Save outputs ---
    results_path = os.path.join(log_dir, "evaluate_global_results.json")
    with open(results_path, "w") as f:
        json.dump({"meta": meta, "global_metrics": global_metrics}, f, indent=2)
    print(f"\n[eval_global] saved full results -> {results_path}")

    confusion_path = os.path.join(log_dir, "evaluate_global_confusion.json")
    with open(confusion_path, "w") as f:
        json.dump({
            "confusion_matrix": global_metrics["confusion_matrix"],
            "labels": global_metrics["labels"],
            "label_names": [id_to_name.get(l, str(l)) for l in global_metrics["labels"]],
        }, f, indent=2)
    print(f"[eval_global] saved confusion matrix -> {confusion_path}")

    per_layer_path = os.path.join(log_dir, "evaluate_per_layer.json")
    with open(per_layer_path, "w") as f:
        json.dump(per_layer_metrics, f, indent=2)
    print(f"[eval_global] saved per-layer metrics -> {per_layer_path}")
    print("[eval_global] done.")


if __name__ == "__main__":
    main()
