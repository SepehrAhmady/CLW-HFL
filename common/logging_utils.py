"""Shared instrumentation utilities.

Part 1 section 0 and Part 3 section 3.2 require REAL measured wall-clock
timings and REAL message sizes logged per round (not assumed/hand-estimated
numbers) -- this is what ultimately backs Figures 5, 6, and 7. A single
shared implementation here guarantees every script measures these the same
way.
"""
import csv
import json
import os
import time
from contextlib import contextmanager

import torch


@contextmanager
def timer():
    """Usage:
        with timer() as t:
            do_work()
        elapsed_seconds = t.elapsed
    """
    class _T:
        elapsed = None

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


def state_dict_size_bytes(state_dict):
    """Real serialized size (bytes) of a model/update state_dict, used as
    the plaintext message-size baseline that PHE/SecAgg overhead is
    compared against in Figure 6. Computed by actually counting tensor
    element_size * numel, not estimated.
    """
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total


def bytes_of_object(obj):
    """Fallback message-size measurement for non-tensor payloads (e.g. a
    list of Paillier ciphertext integers, or a SecAgg masked update) -- we
    serialize to JSON/pickle-friendly form and measure actual bytes rather
    than guessing.
    """
    import pickle
    return len(pickle.dumps(obj))


class RoundLogger:
    """Appends one JSON-lines record per round to a .jsonl file, and keeps
    an in-memory list for an end-of-run summary CSV. Every per-layer
    training script and cloud_aggregate.py use this so every figure/table
    is regenerated from real logged data (Deliverables checklist item 8),
    never hand-drawn numbers.
    """

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.records = []
        # Truncate any previous run's log for this exact path.
        open(self.log_path, "w").close()

    def log(self, record: dict):
        record = dict(record)
        record.setdefault("wall_clock_unix", time.time())
        self.records.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def to_csv(self, csv_path):
        if not self.records:
            return
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        keys = sorted({k for r in self.records for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)
