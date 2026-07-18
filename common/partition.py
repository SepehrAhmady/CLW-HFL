"""Non-IID Dirichlet partitioning of a layer's training subset across its
clients (Part 1, section 1.3).

Standard label-skew Dirichlet partitioning (e.g. Hsu et al. 2019, Yurochkin
et al. 2019 style): for each class, draw a Dirichlet(alpha, ..., alpha) over
the num_clients clients to decide what fraction of that class's samples
each client receives. Smaller alpha => more skewed / non-IID; alpha -> inf
=> uniform / IID-like splits. The default alpha=0.5 with sensitivity checks
at 0.1 and 1.0 are configured centrally in config.yaml.
"""
import numpy as np


def dirichlet_partition(labels, num_clients, alpha, seed=0):
    """Partition sample indices into num_clients non-IID shards.

    Args:
        labels: 1D array-like of local (per-layer) class indices for the
            samples being partitioned (already filtered to this layer's
            owned classes, NOT the global 10-class label).
        num_clients: number of clients for this layer.
        alpha: Dirichlet concentration parameter.
        seed: RNG seed (vary this across the >=3 required seed runs).

    Returns:
        List[np.ndarray] of length num_clients, each containing the sample
        indices (into the original `labels` array) assigned to that client.
        Every input index is assigned to exactly one client.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    num_classes = labels.max() + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        if len(class_idx) == 0:
            continue
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # Convert proportions to integer split points.
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)
        for client_id, idx_chunk in enumerate(splits):
            client_indices[client_id].extend(idx_chunk.tolist())

    client_indices = [np.array(sorted(idxs)) for idxs in client_indices]
    return client_indices


def partition_summary(labels, client_indices, num_classes):
    """Per-client class-count table, useful for logging/sanity-checking how
    non-IID a given (alpha, seed) partition actually turned out, and for the
    alpha-sensitivity check required by Part 1, section 1.3.
    """
    labels = np.asarray(labels)
    table = np.zeros((len(client_indices), num_classes), dtype=int)
    for i, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            continue
        vals, counts = np.unique(labels[idxs], return_counts=True)
        table[i, vals] = counts
    return table
