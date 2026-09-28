# Phase prediction

`run_ph_pred_all_snr.py` trains the realisation-level phase + SNR predictor used by the predicted-phase posterior models.

By default it reads:

```text
data/PIPE_GWs_3PN_PTA_expanded_realizations.npz
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

