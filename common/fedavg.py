"""Weighted FedAvg aggregation of client state_dicts.

Used by every per-layer training script (Part 3, section 3.2: "weighted
aggregation"). This is intentionally a REAL weighted average over actual
client updates -- see Part 2, item 4 for why that distinction matters at
the Cloud level (secure_aggregation must NOT just be this renamed). At the
per-layer level, plain weighted FedAvg is exactly what the spec asks for;
it is cloud_aggregate.py that must go further (FedAdam + SecAgg + DP + PHE).
"""
import torch


def weighted_average_state_dicts(state_dicts, weights):
    """Weighted average of a list of model state_dicts.

    Args:
        state_dicts: list of OrderedDict (torch state_dicts), all with the
            exact same keys/shapes.
        weights: list of non-negative numbers (e.g. per-client sample
            counts), same length as state_dicts. Normalized internally.

    Returns:
        A new state_dict with the same keys, each tensor being the
        weighted average across clients, cast back to its original dtype
        (so integer buffers like BatchNorm's num_batches_tracked stay
        integer -- the weighted average is computed in float and rounded
        via the final dtype cast, which is an accepted approximation for
        that bookkeeping counter specifically).
    """
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total = float(sum(weights))
    assert total > 0, "weighted_average_state_dicts: all weights are zero"
    norm_weights = [w / total for w in weights]

    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        orig_dtype = state_dicts[0][k].dtype
        acc = None
        for sd, w in zip(state_dicts, norm_weights):
            term = sd[k].detach().float() * w
            acc = term if acc is None else acc + term
        avg[k] = acc.to(orig_dtype)
    return avg
