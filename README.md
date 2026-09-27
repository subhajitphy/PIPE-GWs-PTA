# PIPE-GWs-PTA

**Physics-informed phase encodings and simulation-based inference for gravitational waves from eccentric supermassive black-hole binaries in pulsar timing arrays.**

This repository contains the public implementation of the DNF/CNF posterior-inference framework developed for:

> **[Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data](https://arxiv.org/abs/2607.03904)**

The repository includes the code for data generation, phase prediction, hierarchical Transformer conditioning, and discrete/continuous normalizing-flow posterior inference.

**Reserved Zenodo dataset DOI:** `10.5281/zenodo.22972338`

> The Zenodo record is currently a draft. The DOI will become the public dataset DOI when the record is published.

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

### macOS / Linux

```bash
git clone https://github.com/subhajitphy/PIPE-GWs-PTA.git
cd PIPE-GWs-PTA

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
git clone https://github.com/subhajitphy/PIPE-GWs-PTA.git
cd PIPE-GWs-PTA

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The CNF implementation requires `torchdiffeq`, which is included in `requirements.txt`.

---

## Data

The exact synthetic PTA datasets used in the study have been uploaded to a Zenodo draft and will be made publicly available under the reserved DOI:

**Reserved DOI:** `10.5281/zenodo.22972338`

The Zenodo deposit contains:

```text
PIPE_GWs_3PN_PTA_default_realizations.npz
```

Default dataset used for the primary posterior-inference experiments.

```text
PIPE_GWs_3PN_PTA_expanded_realizations.npz
```

Expanded dataset used for the large-data, higher-dimensional, and phase-prediction analyses.

The Zenodo record also contains:

```text
README_DATA.md
```

with dataset-level documentation.

After downloading a dataset, place it inside:

```text
data/
```

or specify its location using the environment variables described below.

---

## Phase-predictor checkpoint

For predicted-phase inference, place the pretrained phase-predictor checkpoint at:

```text
phase_prediction/phase_predictor_best_fast.pt
```

A different checkpoint can be supplied with:

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

## Running the DNF model

The DNF runner is:

```bash
python training/run_dnfs.py
```

The default posterior targets are:

```python
target_names = [
    "log10_n",
    "e0",
    "log10_M",
    "log10_A",
]
```

### No-phase baseline

Set in `training/run_dnfs.py`:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = False
```

Then run:

```bash
python training/run_dnfs.py
```

### Predicted-phase model

Set:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = True
```

Then run:

```bash
python training/run_dnfs.py
```

The pretrained phase predictor is loaded through `PhaseProvider`. The true orbital phase is not supplied to the posterior estimator in this configuration.

---

## Running the CNF model

The CNF runner is:

```bash
python training/run_cnfs.py
```

### No-phase baseline

Set in `training/run_cnfs.py`:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = False
```

Then run:

```bash
python training/run_cnfs.py
```

### Predicted-phase model

Set:

```python
USE_TRUE_PHASE = False
USE_PHASE_PROVIDER = True
```

Then run:

```bash
python training/run_cnfs.py
```

The CNF is trained in fp32 and uses the ODE settings defined directly in `training/run_cnfs.py`.

---

## Selecting a dataset

The runners support environment-variable overrides.

### Default dataset

```bash
PIPE_PTA_DATASET=PIPE_GWs_3PN_PTA_default_realizations.npz \
python training/run_dnfs.py
```

or

```bash
PIPE_PTA_DATASET=PIPE_GWs_3PN_PTA_default_realizations.npz \
python training/run_cnfs.py
```

### Expanded dataset

```bash
PIPE_PTA_DATASET=PIPE_GWs_3PN_PTA_expanded_realizations.npz \
python training/run_dnfs.py
```

or

```bash
PIPE_PTA_DATASET=PIPE_GWs_3PN_PTA_expanded_realizations.npz \
python training/run_cnfs.py
```

### Different local data directory

```bash
PIPE_PTA_DATA_PATH=/path/to/data \
python training/run_dnfs.py
```

The same environment variables can be used with the CNF runner.

---

## Training the phase predictor

The phase-prediction network can be retrained using:

```bash
python phase_prediction/run_ph_pred_all_snr.py
```

The default training setup uses:

```text
SNR range: 10--100
sampling:  log-uniform
```

The model predicts the orbital phase through the circular representation

```text
(cos φ, sin φ)
```

together with a realisation-level SNR diagnostic.

For phase-predictor training, use the expanded dataset:

```text
PIPE_GWs_3PN_PTA_expanded_realizations.npz
```

---

## Regenerating the simulations

The synthetic PTA simulations can be regenerated using:

```bash
python data/gen_data.py
```

Before running the generator, populate:

```text
data/gwecc/
```

with the waveform-generation code and `pulsar_info.csv` used for the study.

The current generator writes the default dataset using its internal filename:

```text
lr_signals_3PN_E_B_phase.npz
```

For the public Zenodo release, the corresponding dataset is archived using the clearer filename:

```text
PIPE_GWs_3PN_PTA_default_realizations.npz
```

The generator stores the simulated timing residuals, orbital phase, source parameters, and metadata required by the downstream inference scripts.

---

## Output files

Training outputs are written to mode-specific directories under:

```text
outputs/
```

The training scripts save model checkpoints, validation information, and training diagnostics.

---

## Reproducibility

The released scripts include the principal settings used for the experiments, including:

- random seed `42`,
- deterministic train/validation splitting,
- fixed noise-generation seeds,
- input/output standardization,
- AdamW optimization,
- learning-rate scheduling,
- early stopping,
- gradient clipping,
- model and preprocessing metadata stored in checkpoints.

Exact numerical agreement can depend on PyTorch, CUDA, GPU, and hardware versions.

---

## Data availability

The exact synthetic PTA datasets supporting the study, including the default and expanded realisation datasets, have been uploaded to a Zenodo draft and will be made publicly available under:

**Reserved DOI:** `10.5281/zenodo.22972338`

The corresponding data-generation, phase-prediction, and posterior-inference code is provided in this repository.

This section will be updated once the Zenodo record is formally published.

---

## License

The software in this repository is released under the **MIT License**.

The associated Zenodo datasets use a separately specified data license.

---

## Citation

The associated manuscript is available as an arXiv preprint:

**S. Dandapat and A. J. K. Chua**,  
*Transformers with Physics-Informed Encodings and Simulation-Based Inference for Robust Detection of Eccentric Binary Black Holes in Pulsar Timing Array Data*,  
[arXiv:2607.03904](https://arxiv.org/abs/2607.03904) (2026).

The journal citation will be added after publication.

Once the Zenodo dataset record is published, its final citation information will also be added here.

---

## Contact

For questions about the code, datasets, or reproducibility of the analysis, please open an issue in this repository.
