# CLW-HFL: Task-Decomposed Hierarchical Federated Learning for Privacy-Preserving Network Intrusion Detection

> IEEE Conference Paper — Sepehr Ahmadi, Mahmood Ahmadi — Razi University, Kermanshah, Iran

## Overview

CLW-HFL is a four-layer hierarchical federated learning (HFL) system for network intrusion detection on the UNSW-NB15 dataset. Its core design principle is **task decomposition by network visibility**: the ten attack categories are split across three specialized federated layers, each matched to the network tier where its assigned attacks are most naturally observable. A Cloud aggregation layer applies a full cryptographic privacy stack on top.

```
Layer A (Edge Nodes, 30 clients)   →  Exploits, Shellcode, Fuzzers, Generic, Normal
Layer B (Routers,   10 clients)    →  Reconnaissance, DoS, Worms, Analysis, Normal  
Layer C (Gateways,  10 clients)    →  Backdoor, Normal
Cloud   (Global Server)            →  SecAgg + Distributed DP + Paillier PHE + FedAdam
```

**Results on UNSW-NB15 (10-class evaluation):**

| Method | Accuracy | F1 (Weighted) | F1 (Macro) | Comm. Volume |
|--------|----------|---------------|------------|--------------|
| Centralized | 0.70 ± 0.007 | 0.74 ± 0.007 | 0.42 ± 0.005 | 100% |
| Flat FedAvg | 0.65 ± 0.008 | 0.69 ± 0.008 | 0.33 ± 0.025 | 12.5% |
| **CLW-HFL** | **0.78** | **0.81** | **0.58** | **9.6%** |

> **Note:** CLW-HFL results use oracle routing by true label (upper bound). See paper for full discussion.

---

## Repository Structure

```
CLW-HFL/
├── config.yaml                  # All hyperparameters (single source of truth)
├── preprocess.py                # UNSW-NB15 preprocessing → data/processed/
├── layer_A_federated.py         # FL training: Edge layer (30 clients, 5-class)
├── layer_B_federated.py         # FL training: Router layer (10 clients, 5-class)
├── layer_C_federated.py         # FL training: Gateway layer (10 clients, binary)
├── cloud_aggregate.py           # Cloud re-aggregation: SecAgg + DP + PHE + FedAdam
├── evaluate_global.py           # Unified 10-class evaluation (oracle routing)
├── centralized_baseline.py      # Centralized baseline (60 epochs, 10-class)
├── flat_fedavg_baseline.py      # Flat FedAvg baseline (50 clients, 15 rounds)
├── run_ablation.py              # Ablation: incremental privacy mechanism study
├── requirements.txt
└── common/
    ├── model.py                 # MultiHeadIDSMLP architecture
    ├── label_utils.py           # Canonical label map + layer-to-class mapping
    ├── focal_loss.py            # Class-Balanced Focal Loss (Cui 2019 + Lin 2017)
    ├── fedavg.py                # Weighted FedAvg aggregation
    ├── fedopt.py                # FedAdam server optimizer
    ├── secure_agg.py            # Pairwise additive secret masking (Bonawitz 2017)
    ├── paillier_utils.py        # Quantize + pack + Paillier PHE pipeline
    ├── dp_accounting.py         # RDP accountant → (ε, δ)-DP conversion
    ├── partition.py             # Dirichlet Non-IID partitioning
    ├── metrics.py               # Accuracy, F1, confusion matrix utilities
    ├── logging_utils.py         # Per-round timing and message-size logging
    └── vector_utils.py          # Flatten/unflatten state_dict ↔ flat vector
```

---

## Setup

**Requirements:** Python 3.12, CUDA optional (CPU-only supported)

```bash
git clone https://github.com/YOUR_USERNAME/CLW-HFL.git
cd CLW-HFL
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

Download **UNSW-NB15** from Kaggle:
👉 https://www.kaggle.com/datasets/dhoogla/unswnb15

Place the files as:
```
CLW-HFL/
└── UNSW_NB15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

---

## Reproducing Results

Run scripts in this order:

**Step 1 — Preprocess:**
```bash
python preprocess.py
```

**Step 2 — Train each layer:**
```bash
python layer_A_federated.py --alpha 0.5 --seed 42
python layer_B_federated.py --alpha 0.5 --seed 42
python layer_C_federated.py --alpha 0.5 --seed 42
```

**Step 3 — Cloud aggregation:**
```bash
python cloud_aggregate.py --seed 42
```

**Step 4 — Global evaluation:**
```bash
python evaluate_global.py
```

**Step 5 — Baselines (3 seeds each):**
```bash
python centralized_baseline.py --epochs 60
python flat_fedavg_baseline.py
```

All results are saved to `logs/` as JSON and CSV.

---

## Privacy Stack (Cloud Layer)

The Cloud aggregation applies five sequential mechanisms to Layer C's client updates:

1. **DP Sensitivity Clipping** — L2 norm clipped to C = 10.0
2. **Distributed Gaussian DP Noise** — σ = noise_multiplier × C, added client-side; RDP accounting tracks cumulative (ε, δ)
3. **Pairwise Additive Secret Masking** — HMAC-SHA256 derived masks cancel in sum; server never sees individual plaintext updates
4. **Quantization + Paillier PHE** — values quantized to 16-bit fixed-point, packed (48 scalars/ciphertext), encrypted under 2048-bit Paillier key; server aggregates homomorphically
5. **FedAdam** — server-side Adam optimizer (ηs=0.1, β1=0.9, β2=0.99)

BatchNorm running statistics are averaged separately via plain FedAvg (excluded from PHE pipeline).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ahmadi2025clwhfl,
  title     = {{CLW-HFL}: Task-Decomposed Hierarchical Federated Learning for Privacy-Preserving Network Intrusion Detection},
  author    = {Ahmadi, Sepehr and Ahmadi, Mahmood},
  booktitle = {Proceedings of the IEEE International Conference on Computer and Knowledge Engineering (ICCKE)},
  year      = {2025},
  address   = {Mashhad, Iran}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
