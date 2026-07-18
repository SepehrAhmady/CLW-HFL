"""Canonical label handling for CLW-HFL.

Per spec Part 1 section 1.2, `label_map.json` is generated ONCE by
preprocess.py (via sklearn.LabelEncoder on attack_cat, alphabetically
sorted) and is the single source of truth for the global 10-class label
space. Every other script must load it from here -- never hardcode a
placeholder mapping (that was the exact bug in the old cloud eval script,
Part 2 item 2).

This module also encodes LAYER_ATTACK_MAP (Part 1, section 1.3): the
intentional, non-overlapping (except for "Normal") assignment of attack
categories to layers.
"""
import json
import os

# Confirmed ground truth (Part 1, section 1.2). preprocess.py must assert
# that the label map it generates from the real data matches this exactly,
# and raise loudly if it does not (e.g. because the wrong CSV/dataset was
# loaded -- see assert_canonical_label_map below).
CANONICAL_LABEL_MAP = {
    "Analysis": 0,
    "Backdoor": 1,
    "DoS": 2,
    "Exploits": 3,
    "Fuzzers": 4,
    "Generic": 5,
    "Normal": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}

# Confirmed ground truth (Part 1, section 1.3). Each attack category is
# handled by exactly one layer; "Normal" appears in every layer's local
# task because every layer needs a negative class to discriminate against.
LAYER_ATTACK_MAP = {
    "A": ["Exploits", "Shellcode", "Fuzzers", "Generic", "Normal"],   # 5-way -> head_binary
    "B": ["Reconnaissance", "DoS", "Worms", "Analysis", "Normal"],    # 5-way -> head_category
    "C": ["Backdoor", "Normal"],                                      # binary -> head_binary
}

# Which head name each layer trains/owns, per Part 6 (the unified eval spec).
# NOTE: the name "head_binary" is reused by both Layer A (5-way) and Layer C
# (2-way) -- this is just an artifact of the shared MultiHeadIDSMLP class
# exposing three generically-named heads (head_binary/head_category/head_fine)
# that get repurposed per layer. It does NOT mean Layer A and Layer C's
# head_binary weights are related or should ever be loaded into each other.
LAYER_HEAD_NAME = {
    "A": "head_binary",
    "B": "head_category",
    "C": "head_binary",
}

CLIENTS_PER_LAYER = {"A": 30, "B": 10, "C": 10}

NUM_GLOBAL_CLASSES = len(CANONICAL_LABEL_MAP)


def load_label_map(path):
    """Load the canonical label_map.json written by preprocess.py.

    Raises FileNotFoundError with an actionable message if preprocessing
    has not been run yet, and ValueError if the loaded map does not match
    the confirmed ground truth (Part 1, section 1.2) -- e.g. because the
    wrong dataset was used to generate it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"label_map.json not found at '{path}'. Run preprocess.py first; "
            "every downstream script depends on this file as the single "
            "source of truth for the label space."
        )
    with open(path, "r") as f:
        label_map = json.load(f)
    assert_canonical_label_map(label_map)
    return label_map


def assert_canonical_label_map(label_map):
    """Sanity check: the generated label map must exactly match Part 1,
    section 1.2. This is the dataset-identity guard described in Part 1,
    section 1.1 -- if this fails, the wrong CSV (e.g. a CSE-CIC-IDS2018
    label file) was almost certainly loaded instead of UNSW-NB15.
    """
    if label_map != CANONICAL_LABEL_MAP:
        raise ValueError(
            "label_map.json does not match the confirmed UNSW-NB15 label "
            f"encoding from Part 1 section 1.2.\nExpected: {CANONICAL_LABEL_MAP}\n"
            f"Got: {label_map}\n"
            "This almost certainly means the wrong dataset/label file was "
            "loaded (the spec explicitly calls out a past incident where a "
            "CSE-CIC-IDS2018 label file leaked into a UNSW-NB15 pipeline). "
            "Re-run preprocess.py against the genuine UNSW-NB15 CSVs."
        )


def layer_classes(layer):
    """Ordered list of attack_cat strings that `layer` ('A'/'B'/'C') owns."""
    return LAYER_ATTACK_MAP[layer]


def build_layer_remap(layer, label_map):
    """Build the two mappings needed to move between a layer's local head
    index space and the canonical global label space.

    Returns:
        global_to_local: dict[int global_label_id -> int local_head_index]
            Only contains entries for classes this layer owns. Local indices
            are assigned in the same order as LAYER_ATTACK_MAP[layer], which
            is also the order used to slice the layer's training subset and
            size its head (len(LAYER_ATTACK_MAP[layer]) == head out_features).
        local_to_global: dict[int local_head_index -> int global_label_id]
            The inverse mapping, used by evaluate_global.py (Part 6, step 3)
            to translate a layer's raw prediction back into the canonical
            10-class space before it is dropped into the global confusion
            matrix.
    """
    classes = layer_classes(layer)
    global_to_local = {label_map[c]: i for i, c in enumerate(classes)}
    local_to_global = {i: label_map[c] for i, c in enumerate(classes)}
    return global_to_local, local_to_global


def layer_num_classes(layer):
    return len(LAYER_ATTACK_MAP[layer])


def filter_and_remap(layer, label_map, x, y_global):
    """Slice (x, y_global) down to the rows this layer owns (Part 1,
    section 1.3) and remap y_global into the layer's own local head index
    space (0..layer_num_classes(layer)-1), in the order given by
    LAYER_ATTACK_MAP[layer].

    Used by every per-layer training script and by evaluate_global.py to
    build each layer's own training/validation/evaluation subset from the
    shared global train.npz/val.npz/test.npz produced by preprocess.py.

    Args:
        layer: 'A', 'B', or 'C'.
        label_map: the canonical label_map.json contents (global label
            space, Part 1 section 1.2).
        x: np.ndarray of shape (N, num_features).
        y_global: np.ndarray of shape (N,), canonical global label ids.

    Returns:
        x_layer: np.ndarray, only rows whose global label is owned by
            this layer.
        y_local: np.ndarray of the same length, remapped to local indices.
    """
    import numpy as np

    global_to_local, _ = build_layer_remap(layer, label_map)
    owned_globals = np.array(sorted(global_to_local.keys()))
    mask = np.isin(y_global, owned_globals)
    x_layer = x[mask]
    y_global_layer = y_global[mask]
    # Vectorized remap via a lookup table sized to the max global label id.
    lut = np.full(int(y_global.max()) + 1, fill_value=-1, dtype=np.int64)
    for g, l in global_to_local.items():
        lut[g] = l
    y_local = lut[y_global_layer]
    assert (y_local >= 0).all(), "filter_and_remap produced an unmapped label -- bug"
    return x_layer, y_local
