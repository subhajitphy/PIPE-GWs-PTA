# Phase prediction

`run_ph_pred_all_snr.py` trains the realisation-level phase + SNR predictor used by the predicted-phase posterior models.

By default it reads:

```text
data/lr_signals_3PN_E_B_phase_expanded.npz
```

and trains over realisation SNR 10--100 with log-uniform SNR sampling.

Run:

```bash
python phase_prediction/run_ph_pred_all_snr.py
```

The best fast checkpoint is written as:

```text
phase_prediction/phase_predictor_best_fast.pt
```

For the public repository, place the paper checkpoint `phase_predictor_best_fast.pt` in this directory. The DNF/CNF runners load it automatically in predicted-phase mode.

To use a different input dataset:

```bash
PIPE_PTA_PHASE_DATASET=my_dataset.npz python phase_prediction/run_ph_pred_all_snr.py
```
