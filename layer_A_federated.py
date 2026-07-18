"""layer_A_federated.py -- CLW-HFL Part 3, section 3.2.

Federated training for Layer A: 30 clients, 5-way `head_binary` over
['Exploits', 'Shellcode', 'Fuzzers', 'Generic', 'Normal'] (Part 1, section
1.3). The actual FL loop lives in common/federated_trainer.py (shared
across Layer A/B/C) -- this script just loads data/config and calls it.

Outputs:
    checkpoints/layer_A_final.pth               (model state_dict)
    checkpoints/layer_A_final_metadata.json      (label remap, classes,
                                                   training time, final
                                                   per-class val metrics)
    logs/layer_A_alpha<a>_seed<s>_rounds.jsonl   (per-round log)
    logs/layer_A_alpha<a>_seed<s>_rounds.csv     (same, as CSV)

Example:
    python layer_A_federated.py --alpha 0.5 --seed 42
    python layer_A_federated.py --alpha 0.1 --seed 13 --rounds 5   # quick smoke test
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.label_utils import load_label_map
from common.federated_trainer import run_layer_training

LAYER = "A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alpha", type=float, default=None,
                         help="Dirichlet alpha override (default: config.yaml "
                              "partitioning.default_alpha=0.5). Use 0.1 or 1.0 "
                              "for the sensitivity check in Part 1, section 1.3.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=None,
                         help="Override number of federated rounds (e.g. for a "
                              "quick smoke test before a full run).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config["runtime"]["device"]
    processed_dir = config["paths"]["processed_dir"]
    label_map = load_label_map(config["paths"]["label_map_path"])

    train_npz = np.load(os.path.join(processed_dir, "train.npz"))
    val_npz = np.load(os.path.join(processed_dir, "val.npz"))

    run_layer_training(
        layer=LAYER,
        config=config,
        x_train_global=train_npz["X"],
        y_train_global=train_npz["y"],
        x_val_global=val_npz["X"],
        y_val_global=val_npz["y"],
        label_map=label_map,
        alpha=args.alpha,
        seed=args.seed,
        rounds_override=args.rounds,
        device=device,
        checkpoint_dir=config["paths"]["checkpoints_dir"],
        log_dir=config["paths"]["logs_dir"],
    )


if __name__ == "__main__":
    main()
