# PIPE-GWs-PTA

**Physics-informed phase encodings and simulation-based inference for gravitational waves from eccentric supermassive black-hole binaries in pulsar timing arrays.**

This repository contains the code supporting the DNF/CNF posterior-inference framework developed for the study:

> **Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data**

The framework combines a hierarchical Transformer conditioner, predicted orbital-phase positional encodings, and conditional discrete/continuous normalizing flows for simulation-based inference (SBI). The phase-conditioned contribution can be restricted to parameters most closely associated with orbital evolution while all inferred parameters remain within a single joint posterior.

A Zenodo dataset record has been reserved for the exact synthetic datasets used in the study:

**Reserved Zenodo DOI:** [10.5281/zenodo.22972338](https://doi.org/10.5281/zenodo.22972338)

---

## Repository layout

```text
PIPE-GWs-PTA/
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
│
├── models/
│   ├── pta_encoder.py
│   ├── phase_predictor.py
│   ├── hierarchical_dnf.py
│   └── hierarchical_cnf.py
│
├── training/
│   ├── run_dnfs.py
│   └── run_cnfs.py
│
├── phase_prediction/
│   ├── run_ph_pred_all_snr.py
│   ├── phase_predictor_best_fast.pt
│   └── README.md
│
├── data/
│   ├── gen_data.py
│   ├── README.md
│   └── gwecc/
│
└── outputs/
```

The `data/gwecc/` directory is intentionally left empty in the public repository. Populate it with the waveform-generation routines and `pulsar_info.csv` used for the study before regenerating the simulations.

---

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/subhajitphy/PIPE-GWs-PTA.git
cd PIPE-GWs-PTA

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The CNF implementation additionally requires `torchdiffeq`, which is included in `requirements.txt`.

---

## Data

The exact synthetic PTA datasets used in the study will be archived on Zenodo under the reserved DOI:

**Reserved Zenodo DOI:** [10.5281/zenodo.22972338](https://doi.org/10.5281/zenodo.22972338)

The release contains two datasets.

### Default dataset

```text
lr_signals_3PN_E_B_phase.npz
```

This dataset is used for the primary DNF/CNF posterior-inference experiments.

### Expanded dataset

```text
lr_signals_with_params_E_B_phase_base.npz
```

This larger dataset is used for the large-data regime, higher-dimensional posterior analysis, and phase-prediction experiments.

After downloading the datasets, place the required NPZ file in `data/`, or provide its location through the environment variables described below.

The simulations can alternatively be regenerated using:

```bash
python data/gen_data.py
```

The data-generation pipeline produces synthetic PTA timing residuals, orbital-phase evolution, source parameters, and associated metadata.

---

## Phase prediction

The phase-prediction network operates on the full PTA realisation and predicts the shared orbital phase of the source together with a realisation-level SNR diagnostic.

The pretrained checkpoint used by the posterior-inference models is located at:

```text
phase_prediction/phase_predictor_best_fast.pt
```

To retrain the phase-prediction network:

```bash
python phase_prediction/run_ph_pred_all_snr.py
```

The default phase-prediction setup uses an SNR range of 10--100 with log-uniform SNR sampling.

The orbital phase is represented through

```text
(cos φ, sin φ)
```

rather than regressing directly on a wrapped angular variable.

---

## Posterior inference

Posterior inference is implemented using:

- **DNF:** a conditional discrete normalizing flow based on affine coupling transformations.
- **CNF:** a conditional continuous normalizing flow based on neural ODE evolution.

Both approaches use the same hierarchical Transformer conditioning architecture and support the phase-agnostic and predicted-phase configurations compared in the study.

---

## 1. No-phase baseline

For the phase-agnostic baseline, set in either `training/run_dnfs.py` or `training/run_cnfs.py`:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = False
```

This gives:

```python
USE_PHASE = False
```

The posterior estimator is then conditioned only on the base hierarchical Transformer representation.

Run either:

```bash
python training/run_dnfs.py
```

or

```bash
python training/run_cnfs.py
```

---

## 2. Predicted-phase PIPE model

For predicted-phase conditioning, set:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = True
```

This gives:

```python
USE_PHASE = True
```

The pretrained phase predictor is loaded through `PhaseProvider`.

The true orbital phase is **not** supplied to the posterior estimator in this configuration. Instead, the phase is inferred from the noisy PTA realisation and then used to construct the physics-informed phase encoding.

Run either:

```bash
python training/run_dnfs.py
```

or

```bash
python training/run_cnfs.py
```

Training outputs, checkpoints, and diagnostics are written to mode-specific output directories.

---

## Hierarchical Transformer conditioner

The posterior-inference models use a hierarchical Transformer architecture.

Each pulsar time series is first processed independently using:

- 1D patch embeddings,
- sinusoidal positional encoding,
- optional physics-informed phase encoding,
- standard multi-head self-attention.

The resulting pulsar-level representations are then processed using cross-pulsar self-attention to capture array-level correlations.

The final pooled representation provides the conditioning context for the DNF or CNF posterior estimator.

---

## Physics-informed phase conditioning

The posterior architecture uses separate base and phase-conditioned representations:

```text
h_base  = E_base(x)
h_phase = E_phase(x, φ)
h_delta = h_phase - h_base
```

The base representation is always supplied to the posterior estimator.

When predicted-phase conditioning is enabled, the additional phase-dependent contribution `h_delta` is injected only into selected parameter dimensions. This allows the model to exploit explicit orbital-phase information while preserving a single joint posterior over all inferred parameters.

---

## Default 4D posterior experiment

The default posterior experiment infers:

```python
target_names = [
    "log10_n",
    "e0",
    "log10_M",
    "log10_A",
]
```

The direct phase-conditioned contribution is restricted to:

```python
phase_target_names = [
    "log10_n",
    "e0",
    "log10_M",
]
```

The amplitude parameter `log10_A` remains part of the same joint posterior but does not receive a direct phase-conditioned correction.

---

## Masked phase conditioning

For the DNF, phase conditioning modifies only the selected transformed coordinates of each affine coupling layer.

Schematically,

```text
(s, t)
=
(s_base, t_base)
+
m_phi ⊙ (Δs_phi, Δt_phi).
```

For the CNF, the phase-dependent correction is applied to selected components of the continuous vector field:

```text
dθ/dt
=
f_base
+
g_phi m_phi ⊙ f_phi.
```

Here:

- `m_phi` denotes the phase-target mask,
- `g_phi` is an explicit phase gate,
- all inferred parameters remain within the same joint flow.

---

## Higher-dimensional experiment

The synthetic datasets also contain the broader EBBH source-parameter vector:

```python
target_names = [
    "cos_gwtheta",
    "gwphi",
    "psi",
    "cos_inc",
    "log10_n",
    "q",
    "e0",
    "log10_M",
    "log10_A",
]
```

For the higher-dimensional analysis, the direct phase-conditioned subset remains:

```python
phase_target_names = [
    "log10_n",
    "e0",
    "log10_M",
]
```

The expanded dataset should be used for the corresponding large-data analysis.

---

## Selecting a different dataset

The training scripts support environment-variable overrides so that machine-specific paths do not need to be hard-coded.

To select a dataset:

```bash
PIPE_PTA_DATASET=lr_signals_3PN_E_B_phase.npz \
python training/run_dnfs.py
```

or

```bash
PIPE_PTA_DATASET=lr_signals_3PN_E_B_phase.npz \
python training/run_cnfs.py
```

A different local data directory can be supplied using:

```bash
PIPE_PTA_DATA_PATH=/path/to/data \
python training/run_dnfs.py
```

For the expanded dataset:

```bash
PIPE_PTA_DATASET=lr_signals_with_params_E_B_phase_base.npz \
python training/run_dnfs.py
```

---

## Selecting the phase-predictor checkpoint

A different phase-predictor checkpoint can be supplied using:

```bash
PIPE_PTA_PHASE_CKPT=/path/to/phase_predictor_best_fast.pt \
python training/run_dnfs.py
```

or

```bash
PIPE_PTA_PHASE_CKPT=/path/to/phase_predictor_best_fast.pt \
python training/run_cnfs.py
```

---

## Reproducibility

The released scripts contain the principal settings required to reproduce the experiments.

Key reproducibility settings include:

- global random seed: `42`,
- deterministic train/validation splitting,
- deterministic noise generation with fixed seeds,
- fixed model hyperparameters,
- AdamW optimization,
- learning-rate scheduling,
- early stopping based on validation loss,
- gradient clipping,
- stored input/output standardization statistics.

### DNF

The DNF implementation uses conditional affine coupling transformations with a hierarchical self-attention Transformer conditioner.

Training settings are defined in:

```text
training/run_dnfs.py
```

### CNF

The CNF implementation uses neural ODE evolution and evaluates the divergence exactly in the low-dimensional posterior space.

The released configuration uses:

```text
ODE solver:       RK4
absolute tol.:    1e-3
relative tol.:    1e-3
step size:        0.1
training dtype:   fp32
```

Training settings are defined in:

```text
training/run_cnfs.py
```

### Attention

The released hierarchical PTA encoder uses standard multi-head self-attention for both:

1. per-pulsar temporal processing, and
2. cross-pulsar aggregation.

No external-attention module is used in the released architecture.

---

## Checkpoints

Posterior checkpoints store information including:

- model state,
- optimizer state,
- training epoch,
- validation performance,
- target parameter names,
- phase-target indices,
- data-standardization statistics,
- model configuration metadata.

The pretrained phase-prediction checkpoint additionally stores the normalization statistics required for phase inference.

Exact numerical agreement can depend on the PyTorch, CUDA, GPU, and hardware environment.

---

## Data availability

The exact synthetic PTA datasets supporting the study, including the default and expanded realisation datasets, are being prepared for public release on Zenodo under:

**Reserved DOI:** [10.5281/zenodo.22972338](https://doi.org/10.5281/zenodo.22972338)

The source code for data generation, phase prediction, and DNF/CNF posterior inference is provided in this repository.

This section will be updated when the Zenodo dataset record is formally published.

---

## License

The software in this repository is released under the **MIT License**.

The associated Zenodo datasets are distributed under their separately specified data license.

---

## Citation

Citation information for the associated paper will be added after acceptance/publication.

---

## Contact

For questions about the code, datasets, or reproducibility of the analysis, please open an issue in this repository.
