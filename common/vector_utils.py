"""Flatten/unflatten a torch state_dict <-> a single flat numpy vector.

Secure aggregation (pairwise masking), DP noising, and Paillier
quantize+pack all operate most naturally on one flat vector per client
update rather than per-tensor. This module is the single place that
defines that flattening convention, used throughout cloud_aggregate.py.
"""
import numpy as np
import torch

# Keys that contain BatchNorm running statistics / counters.
# These are NOT learnable parameters -- they are computed statistics that
# accumulate during forward passes in training mode. Including them in
# the FedAdam+quantization+PHE pipeline corrupts them (they go through
# floating-point->int->float round-trips and Adam dynamics that are not
# designed for running stats), which makes eval-mode performance collapse
# (BatchNorm in eval mode uses these running stats, not batch stats).
# Fix: separate them out, average them independently (no encryption needed),
# and merge back after FedAdam. See cloud_aggregate.py.
BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def is_bn_buffer(key: str) -> bool:
    return any(key.endswith(s) for s in BN_BUFFER_SUFFIXES)


def split_trainable_and_bn_buffers(state_dict):
    """Split a state_dict into:
        trainable: dict of learnable parameters (go through FedAdam+PHE)
        bn_buffers: dict of BatchNorm running stats (averaged separately)
    """
    trainable = {k: v for k, v in state_dict.items() if not is_bn_buffer(k)}
    bn_buffers = {k: v for k, v in state_dict.items() if is_bn_buffer(k)}
    return trainable, bn_buffers


def average_bn_buffers(per_client_bn_buffers, weights):
    """Weighted average of BatchNorm running stats across clients.
    Straightforward numpy weighted average -- no encryption, no DP noise.
    Returns a dict with the same keys as the input dicts.
    """
    total = float(sum(weights))
    out = {}
    keys = per_client_bn_buffers[0].keys()
    for k in keys:
        acc = None
        for client_bufs, w in zip(per_client_bn_buffers, weights):
            arr = client_bufs[k].detach().cpu().float().numpy() * (w / total)
            acc = arr if acc is None else acc + arr
        # num_batches_tracked is an integer counter -- round to nearest int.
        if k.endswith("num_batches_tracked"):
            out[k] = torch.from_numpy(np.atleast_1d(np.round(acc).astype(np.int64))).squeeze()
        else:
            out[k] = torch.from_numpy(np.atleast_1d(acc.astype(np.float32))).squeeze()
    return out


def flatten_state_dict(state_dict):
    """Returns (flat_vector: np.ndarray[float64], layout: list of (key, shape, numel))
    in a fixed, deterministic key order (dict insertion order, which for a
    state_dict loaded from active_state_dict() is stable across clients of
    the same layer).
    """
    layout = []
    parts = []
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy().astype(np.float64).reshape(-1)
        layout.append((k, tuple(v.shape), arr.size))
        parts.append(arr)
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    return flat, layout


def unflatten_to_state_dict(flat_vector, layout, dtypes=None):
    """Inverse of flatten_state_dict. `dtypes` is an optional dict
    key->torch.dtype to cast each tensor back to (defaults to float32).
    """
    out = {}
    offset = 0
    for key, shape, numel in layout:
        chunk = flat_vector[offset:offset + numel]
        offset += numel
        t = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).reshape(shape)
        if dtypes is not None and key in dtypes:
            t = t.to(dtypes[key])
        out[key] = t
    assert offset == len(flat_vector), "layout/flat_vector length mismatch"
    return out