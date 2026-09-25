# Data

This directory is the local data location used by the training scripts.

The exact datasets used in the paper are intended to be archived on Zenodo. Before public release, add the Zenodo DOI/link here.

Expected filenames:

- `lr_signals_3PN_E_B_phase.npz` — default dataset (50,000 accepted PTA realisations by default).
- `lr_signals_3PN_E_B_phase_expanded.npz` — expanded dataset used for the large-data / higher-dimensional analysis and phase-prediction training.

The scripts expect each NPZ to contain at least:

- `X_B`: PTA residuals, shape `(P, R, L)`.
- `phase_B`: shared orbital phase, shape `(R, L)`.
- `Y_by_pulsar`: source parameters for each pulsar/realisation.
- `param_cols`: names of the stored source parameters.

## Regenerating data

`gen_data.py` provides the data-generation pipeline. The `gwecc/` directory is intentionally left empty in this archive; populate it with the waveform package and `pulsar_info.csv` used for the study before running the generator.

Default 50k dataset:

```bash
python data/gen_data.py
```

Expanded 400k dataset:

```bash
PIPE_PTA_N_TARGET=400000 \
PIPE_PTA_OUTPUT=lr_signals_3PN_E_B_phase_expanded.npz \
python data/gen_data.py
```

The generator uses seed 42 unless changed in `gen_data.py`.
