#!/usr/bin/env python
# coding: utf-8
"""
Generate PTA residual datasets with:
- Fixed shape outputs: X_E, X_B: (10, N, t_step) with NO NaN holes
- Shared "base" parameters across pulsars (same intrinsic/extrinsic GW params),
  only pulsar (theta, phi, pdist) differ per pulsar.
- phase_B: (N, t_step) computed once per base sample
- Y_by_pulsar: (10, N, nparams) with pulsar_idx included

Key idea to avoid NaNs:
✅ Accept a base sample ONLY if:
   (1) phase is finite, AND
   (2) signals for ALL 10 pulsars are finite (Earth and Both), with correct shape.
Otherwise reject and resample.

This guarantees NO NaNs in saved arrays.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

# ===================== Paths / Imports =====================
PKG_PATH = "./gwecc/"
sys.path.append(PKG_PATH)

psr_info = pd.read_csv(f"{PKG_PATH}/pulsar_info.csv")  # needs 'theta','phi','pdist'
from constants import *  # yr, pc, etc.
#from gwecc_res_min import add_ecc_cgw, get_phi
from gwecc_res import add_ecc_cgw, get_phi
# ===================== Config =====================
N_TARGET  = int(5e4)     # per base sample count (same across 10 pulsars)
t_step    = 400
idx_values = np.arange(10, dtype=np.int64)  # pulsar indices 0..9
order     = 3
seed      = 42

# time array
tarr = np.linspace(0, 12 * yr, t_step).astype(np.float64)
t0 = 0.0
gamma_0 = 0.0
l_0 = 0.0

# parallelism
n_jobs = max(1, mp.cpu_count())
backend = "loky"

# batch controls (tune!)
BATCH_DRAW = 2000   # draw this many candidate base samples per iteration
# NOTE: for huge N (1e5), you'll want larger BATCH_DRAW (5k-20k) if memory allows.

# ===================== Pulsar meta =====================
theta_arr = psr_info["theta"].to_numpy(dtype=np.float64)
phi_arr   = psr_info["phi"].to_numpy(dtype=np.float64)
pdist_arr = (psr_info["pdist"].to_numpy(dtype=np.float64) * 1000.0 * pc)  # kpc -> pc -> SI

# ===================== Helpers =====================
def q_to_eta(q: float) -> float:
    """Symmetric mass ratio from mass ratio q = m2/m1 ≤ 1."""
    return q / (1.0 + q)**2

def draw_base_params(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw n candidate base samples."""
    ranges = {
        "log10_n":     (-8.0, -6.7),
        "e0":          (0.1, 0.8),
        "log10_M":     (7.0, 10.0),
        "log10_A":     (-8.0, -6.0),
        "cos_gwtheta": (0.0, 1.0),
        "gwphi":       (0.0, 2*np.pi),
        "q":           (0.1, 1.0),
        "cos_inc":     (0.0, 1.0),
        "psi":         (0.0, np.pi),
    }
    base = {k: rng.uniform(lo, hi, size=n) for k, (lo, hi) in ranges.items()}
    return pd.DataFrame(base)

def compute_phase(row: pd.Series):
    """Compute base phase; return float32 (t_step,) or None if invalid."""
    try:
        n0  = 10.0**float(row["log10_n"])
        e0  = float(row["e0"])
        M   = 10.0**float(row["log10_M"])
        eta = q_to_eta(float(row["q"]))

        z0 = [n0, e0, gamma_0]
        phi = get_phi(tarr, M, z0, eta)
        phi = np.unwrap(phi).astype(np.float32)

        if phi.shape != (t_step,) or (not np.isfinite(phi).all()):
            return None
        return phi
    except Exception:
        return None

def generate_all_pulsars_signals(row: pd.Series):
    """
    For a single base sample (row), generate signals for all 10 pulsars.
    Returns:
      (XE, XB) each (10, t_step) float32 if ALL finite,
      else None
    """
    # extract shared GW params
    try:
        shared = dict(
            cos_gwtheta=float(row["cos_gwtheta"]),
            gwphi=float(row["gwphi"]),
            psi=float(row["psi"]),
            cos_inc=float(row["cos_inc"]),
            log10_n=float(row["log10_n"]),
            q=float(row["q"]),
            e0=float(row["e0"]),
            log10_M=float(row["log10_M"]),
            log10_A=float(row["log10_A"]),
        )
    except Exception:
        return None

    XE = np.empty((10, t_step), dtype=np.float32)
    XB = np.empty((10, t_step), dtype=np.float32)

    for p in idx_values:
        try:
            kw = dict(
                theta=float(theta_arr[p]),
                phi=float(phi_arr[p]),
                pdist=float(pdist_arr[p]),
                **shared
            )

            resE = add_ecc_cgw(
                toas=tarr, tref=t0, gamma_0=gamma_0, l_0=l_0,
                **kw, res="Earth"
            )
            resB = add_ecc_cgw(
                toas=tarr, tref=t0, gamma_0=gamma_0, l_0=l_0,
                **kw, res="Both"
            )

            resE = np.asarray(resE, dtype=np.float32)
            resB = np.asarray(resB, dtype=np.float32)

            if resE.shape != (t_step,) or resB.shape != (t_step,):
                return None
            if (not np.isfinite(resE).all()) or (not np.isfinite(resB).all()):
                return None

            XE[p, :] = resE
            XB[p, :] = resB

        except Exception:
            return None

    return XE, XB

# ===================== Pre-allocate FINAL arrays (NO NaNs) =====================
X_E = np.empty((10, N_TARGET, t_step), dtype=np.float32)
X_B = np.empty((10, N_TARGET, t_step), dtype=np.float32)
phase_B = np.empty((N_TARGET, t_step), dtype=np.float32)

param_cols = [
    "pulsar_idx",
    "cos_gwtheta", "gwphi", "psi", "cos_inc",
    "log10_n", "q", "e0", "log10_M", "log10_A",
]
Y_by_pulsar = np.empty((10, N_TARGET, len(param_cols)), dtype=np.float32)

# keep base params (for reference)
base_rows_kept = []

# ===================== Main accept/reject loop (batched) =====================
rng = np.random.default_rng(seed)
filled = 0

pbar = tqdm(total=N_TARGET, desc="Accepted base samples", ncols=100)

while filled < N_TARGET:
    # ---- 1) draw a batch of candidate base samples ----
    base_df = draw_base_params(rng, BATCH_DRAW)

    # ---- 2) compute phase in parallel, discard invalid ----
    phase_list = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(compute_phase)(base_df.iloc[i])
        for i in range(len(base_df))
    )
    phase_ok = np.array([ph is not None for ph in phase_list], dtype=bool)
    if phase_ok.sum() == 0:
        continue

    base_df_phase = base_df.loc[phase_ok].reset_index(drop=True)
    phase_df = [ph for ph in phase_list if ph is not None]  # aligned with base_df_phase

    # ---- 3) for phase-valid candidates, generate ALL pulsar signals ----
    # Parallelize over candidates (each candidate loops p=0..9 internally)
    sig_list = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(generate_all_pulsars_signals)(base_df_phase.iloc[i])
        for i in range(len(base_df_phase))
    )

    # ---- 4) accept those with valid signals for all pulsars ----
    for i, out in enumerate(sig_list):
        if out is None:
            continue

        XE10, XB10 = out  # each (10,t_step)
        ph = phase_df[i]  # (t_step,)

        # write into final arrays
        j = filled
        X_E[:, j, :] = XE10
        X_B[:, j, :] = XB10
        phase_B[j, :] = ph

        row = base_df_phase.iloc[i]
        base_rows_kept.append(row.to_dict())

        # fill Y_by_pulsar for each pulsar
        shared_vals = np.array([
            float(row["cos_gwtheta"]),
            float(row["gwphi"]),
            float(row["psi"]),
            float(row["cos_inc"]),
            float(row["log10_n"]),
            float(row["q"]),
            float(row["e0"]),
            float(row["log10_M"]),
            float(row["log10_A"]),
        ], dtype=np.float32)

        for p in idx_values:
            Y_by_pulsar[p, j, 0] = float(p)  # pulsar_idx
            Y_by_pulsar[p, j, 1:] = shared_vals

        filled += 1
        pbar.update(1)

        if filled >= N_TARGET:
            break

pbar.close()

# ===================== Final sanity checks =====================
assert np.isfinite(X_E).all()
assert np.isfinite(X_B).all()
assert np.isfinite(phase_B).all()
assert np.isfinite(Y_by_pulsar).all()

print("DONE")
print("X_E:", X_E.shape)
print("X_B:", X_B.shape)
print("phase_B:", phase_B.shape)
print("Y_by_pulsar:", Y_by_pulsar.shape)

# ===================== Save =====================
base_df_kept = pd.DataFrame(base_rows_kept)

out_name = "lr_signals_3PN_E_B_phase.npz"
np.savez(
    out_name,
    X_E=X_E,
    X_B=X_B,
    phase_B=phase_B,
    Y_by_pulsar=Y_by_pulsar,
    base_df=base_df_kept.to_records(index=False),
    param_cols=np.array(param_cols, dtype=object),
    meta=np.array(
        [f"N_TARGET={N_TARGET}", f"t_step={t_step}", f"seed={seed}", f"order={order}"],
        dtype=object
    )
)
print("Saved:", out_name)

