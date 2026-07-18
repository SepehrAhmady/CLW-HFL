"""Shared single-layer federated training engine.

layer_A_federated.py, layer_B_federated.py, and layer_C_federated.py are
thin CLI wrappers around `run_layer_training` below. The spec (Deliverables
checklist) asks for three separate per-layer scripts, but the actual FL
loop -- Dirichlet partitioning, local client training with Class-Balanced
Focal Loss + gradient clipping, weighted FedAvg aggregation, per-round
timing/message-size/val-metric logging, final checkpoint + metadata -- is
identical across layers except for which classes/clients/rounds apply. A
single shared implementation avoids three copies of the same logic quietly
drifting apart (exactly the kind of duplication that produces bugs like
the ones catalogued in Part 2).
"""
import copy
import json
import os
import time

import numpy as np
import torch

from common.label_utils import (
    LAYER_HEAD_NAME,
    build_layer_remap,
    filter_and_remap,
    layer_num_classes,
)
from common.model import build_model_for_layer
from common.partition import dirichlet_partition
from common.focal_loss import ClassBalancedFocalLoss
from common.fedavg import weighted_average_state_dicts
from common.metrics import compute_metrics
from common.logging_utils import RoundLogger, state_dict_size_bytes, timer


def _local_train_one_client(model, x, y, optimizer_cfg, batch_size, local_epochs,
                             grad_clip_norm, head_name, loss_fn, device):
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optimizer_cfg["lr"],
        momentum=optimizer_cfg.get("momentum", 0.0),
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),
    )
    n = len(x)
    for _ in range(local_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) <= 1:
                # BatchNorm1d (inside ResidualBlock) requires >1 sample per
                # batch in training mode. With small per-client datasets
                # (common under Dirichlet skew, especially low alpha), a
                # trailing batch of size 1 is a realistic occurrence, not an
                # edge case -- skip it rather than crash. Worst case, a
                # near-empty client contributes an unchanged copy of the
                # weights it started the round with, which FedAvg handles
                # gracefully via its sample-count weighting.
                continue
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb, head=head_name)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
    return model


@torch.no_grad()
def evaluate_on_subset(model, x, y, head_name, num_classes, device, batch_size=1024):
    if len(x) == 0:
        return None
    model.eval()
    preds = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        logits = model(xb, head=head_name)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    return compute_metrics(y.cpu().numpy(), preds, labels=list(range(num_classes)))


def run_layer_training(layer, config, x_train_global, y_train_global,
                        x_val_global, y_val_global, label_map,
                        alpha=None, seed=42, rounds_override=None,
                        device="cpu", checkpoint_dir="checkpoints", log_dir="logs",
                        verbose=True):
    """Runs the full federated training loop for one layer ('A', 'B', or 'C')
    and writes a checkpoint + metadata file + round-by-round log.

    Returns the path to the saved checkpoint and the metadata dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    head_name = LAYER_HEAD_NAME[layer]
    num_classes = layer_num_classes(layer)
    global_to_local, local_to_global = build_layer_remap(layer, label_map)

    x_train, y_train = filter_and_remap(layer, label_map, x_train_global, y_train_global)
    x_val, y_val = filter_and_remap(layer, label_map, x_val_global, y_val_global)

    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    num_clients = config["clients"][f"layer_{layer}"]
    alpha = config["partitioning"]["default_alpha"] if alpha is None else alpha
    client_indices = dirichlet_partition(y_train, num_clients, alpha, seed=seed)

    # Class-Balanced Focal Loss weights are computed from this layer's own
    # (post-partition-irrelevant) training-label distribution -- the overall
    # class frequency this layer's model will actually be trained against.
    class_counts = np.bincount(y_train, minlength=num_classes)
    loss_fn = ClassBalancedFocalLoss(
        class_counts,
        beta=config["training"]["focal_loss"]["beta"],
        gamma=config["training"]["focal_loss"]["gamma"],
    )

    model = build_model_for_layer(
        layer,
        input_dim=config["model"]["input_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
    ).to(device)

    num_rounds = rounds_override or config["training"]["rounds"][f"layer_{layer}"]
    batch_size = config["training"]["batch_size"]
    local_epochs = config["training"]["local_epochs"]
    grad_clip_norm = config["training"]["grad_clip_norm"]
    optimizer_cfg = config["training"]["optimizer"]
    client_fraction = config["clients"].get("client_fraction_per_round", 1.0)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"layer_{layer}_alpha{alpha}_seed{seed}_rounds.jsonl")
    logger = RoundLogger(log_path)

    rng = np.random.RandomState(seed)
    training_start = time.perf_counter()

    if verbose:
        print(f"[layer_{layer}] {num_clients} clients, {num_rounds} rounds, "
              f"alpha={alpha}, seed={seed}, {len(x_train)} local train rows, "
              f"{len(x_val)} local val rows, classes={list(global_to_local.keys())}")

    for round_idx in range(num_rounds):
        n_selected = max(1, int(round(client_fraction * num_clients)))
        selected = rng.choice(num_clients, size=n_selected, replace=False)

        client_state_dicts, client_weights, client_records = [], [], []
        for cid in selected:
            idxs = client_indices[cid]
            if len(idxs) == 0:
                continue  # extreme non-IID skew can leave a client empty
            local_model = copy.deepcopy(model)
            with timer() as t:
                _local_train_one_client(
                    local_model, x_train_t[idxs], y_train_t[idxs],
                    optimizer_cfg, batch_size, local_epochs, grad_clip_norm,
                    head_name, loss_fn, device,
                )
            # Only the backbone + this layer's active head are a realistic
            # transmission payload -- the other two heads are never trained
            # for this layer and a real client would not send them. This is
            # what both the message-size log and the aggregation step use.
            payload = local_model.active_state_dict(head_name)
            msg_bytes = state_dict_size_bytes(payload)
            client_state_dicts.append(payload)
            client_weights.append(len(idxs))
            client_records.append({
                "client_id": int(cid),
                "num_samples": int(len(idxs)),
                "compute_time_sec": t.elapsed,
                "message_size_bytes": msg_bytes,
            })

        if not client_state_dicts:
            continue  # nothing to aggregate this round (shouldn't normally happen)

        aggregated = weighted_average_state_dicts(client_state_dicts, client_weights)
        # strict=False is intentional: `aggregated` only contains backbone +
        # this layer's active head keys (see payload above), so the other
        # two (inert, never-trained) heads are deliberately left untouched.
        model.load_state_dict(aggregated, strict=False)

        val_metrics = evaluate_on_subset(model, x_val_t, y_val_t, head_name, num_classes, device)

        record = {
            "layer": layer,
            "round": round_idx,
            "alpha": alpha,
            "seed": seed,
            "num_clients_this_round": len(client_records),
            "clients": client_records,
            "total_message_bytes_this_round": sum(c["message_size_bytes"] for c in client_records),
            "max_client_compute_time_sec": max(c["compute_time_sec"] for c in client_records),
            "val_accuracy": val_metrics["accuracy"] if val_metrics else None,
            "val_f1_macro": val_metrics["f1_macro"] if val_metrics else None,
        }
        logger.log(record)

        if verbose and (round_idx % max(1, num_rounds // 10) == 0 or round_idx == num_rounds - 1):
            print(f"[layer_{layer}] round {round_idx + 1}/{num_rounds} "
                  f"val_acc={record['val_accuracy']:.4f} val_f1_macro={record['val_f1_macro']:.4f}")

    total_training_time = time.perf_counter() - training_start
    logger.to_csv(os.path.join(log_dir, f"layer_{layer}_alpha{alpha}_seed{seed}_rounds.csv"))

    final_val_metrics = evaluate_on_subset(model, x_val_t, y_val_t, head_name, num_classes, device)

    ckpt_name = f"layer_{layer}_final.pth"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)

    metadata = {
        "layer": layer,
        "head_name": head_name,
        "classes": list(global_to_local.keys()),
        "global_to_local": {int(k): int(v) for k, v in global_to_local.items()},
        "local_to_global": {int(k): int(v) for k, v in local_to_global.items()},
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "num_rounds": num_rounds,
        "total_training_time_sec": total_training_time,
        "final_val_metrics": final_val_metrics,
    }
    meta_path = os.path.join(checkpoint_dir, f"layer_{layer}_final_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"[layer_{layer}] done in {total_training_time:.1f}s -> {ckpt_path}")
        print(f"[layer_{layer}] final val_accuracy={final_val_metrics['accuracy']:.4f} "
              f"val_f1_macro={final_val_metrics['f1_macro']:.4f}")

    return ckpt_path, metadata
