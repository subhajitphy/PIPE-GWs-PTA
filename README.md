# PIPE-PTA

**Physics-informed phase encodings + simulation-based inference for eccentric binary black holes in pulsar timing arrays.**

This repository contains the code supporting the DNF/CNF posterior-inference framework described in:

> **Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data**

The implementation combines a hierarchical Transformer conditioner, predicted orbital-phase positional encodings, and conditional discrete/continuous normalizing flows. The phase-conditioned contribution can be restricted to the orbital-evolution-sensitive parameters while all parameters remain in one joint posterior.

## Repository layout

```text
PIPE-PTA/
├── README.md
├── requirements.txt
├── CITATION.cff
├── models/
│   ├── pta_encoder.py
│   ├── phase_predictor.py
│   ├── hierarchical_dnf.py
│   └── hierarchical_cnf.py
├── training/
│   ├── run_dnfs.py
│   └── run_cnfs.py
├── phase_prediction/
│   ├── run_ph_pred_all_snr.py
│   ├── phase_predictor_best_fast.pt   # add paper checkpoint
│   └── README.md
├── data/
│   ├── gen_data.py
│   ├── README.md
│   └── gwecc/                         # intentionally blank; populate before use
└── outputs/
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The CNF implementation additionally uses `torchdiffeq` (included in `requirements.txt`).

## Data

The exact default and expanded datasets used in the paper will be archived separately on **Zenodo**. Place the downloaded NPZ files in `data/` using the names described in `data/README.md`.

The simulations can alternatively be regenerated from source with:

```bash
python data/gen_data.py
```

The `data/gwecc/` folder is intentionally blank in this archive. Populate it with the waveform-generation code and `pulsar_info.csv` used for the study before regenerating data.

## Phase predictor

The predicted-phase model operates on the full PTA realisation and predicts the shared orbital phase (and a realisation-level SNR diagnostic). The pretrained paper checkpoint should be placed at:

```text
phase_prediction/phase_predictor_best_fast.pt
```

To retrain it from the expanded dataset:

```bash
python phase_prediction/run_ph_pred_all_snr.py
```

The default training setup uses SNR 10--100 with log-uniform SNR sampling.

## Posterior inference

Both `training/run_dnfs.py` and `training/run_cnfs.py` support the two configurations compared in the paper.

### 1. No-phase baseline

Set in the chosen runner:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = False
```

This gives `USE_PHASE = False`; the posterior is conditioned only on the base hierarchical Transformer representation.

### 2. Predicted-phase PIPE model

Set:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = True
```

This gives `USE_PHASE = True` and loads `phase_prediction/phase_predictor_best_fast.pt` through `PhaseProvider`. The true orbital phase is **not** exposed to the posterior estimator.

Then run either:

```bash
python training/run_dnfs.py
```

or

```bash
python training/run_cnfs.py
```

Outputs are written under `outputs/` in mode-specific directories.

## Default 4D posterior experiment

The current runners infer

```python
target_names = ["log10_n", "e0", "log10_M", "log10_A"]
```

with direct phase-conditioned corrections restricted to

```python
phase_target_names = ["log10_n", "e0", "log10_M"]
```

The DNF applies this mask to the phase-dependent affine-coupling correction; the CNF applies it to the phase-dependent vector-field contribution.

## Full higher-dimensional experiment

For the full EBBH source-parameter analysis, set `target_names` to the full parameter vector stored by `gen_data.py`, for example:

```python
target_names = [
    "cos_gwtheta", "gwphi", "psi", "cos_inc",
    "log10_n", "q", "e0", "log10_M", "log10_A",
]
phase_target_names = ["log10_n", "e0", "log10_M"]
```

Use the expanded dataset from Zenodo for the large-data analysis.

## Selecting a different dataset or checkpoint

The repository-ready runners support environment overrides without editing machine-specific paths:

```bash
PIPE_PTA_DATASET=my_dataset.npz python training/run_dnfs.py
```

```bash
PIPE_PTA_PHASE_CKPT=/path/to/phase_predictor_best_fast.pt \
python training/run_cnfs.py
```

A different local data directory can be supplied with `PIPE_PTA_DATA_PATH`.

## Reproducibility notes

- Global random seed: `42` in the released training/data-generation scripts.
- DNF posterior training: AdamW, deterministic split/noise generation, early stopping, and learning-rate scheduling are defined in `training/run_dnfs.py`.
- CNF posterior training: fp32 ODE training with exact divergence and the RK4 solver settings used in the study are defined in `training/run_cnfs.py`.
- The hierarchical encoder uses standard multi-head self-attention for both per-pulsar temporal processing and cross-pulsar aggregation.
- Checkpoints store model state and data-standardization statistics used by posterior training.

Exact numerical agreement can still depend on PyTorch/CUDA/hardware versions. For archival reproduction, record the environment used for the final paper run together with the Zenodo release.

## Data availability

The exact synthetic datasets supporting the paper (default and expanded realisation sets) are intended to be deposited on Zenodo. Add the DOI here when the deposit is finalized.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated paper if you use this code or data.
