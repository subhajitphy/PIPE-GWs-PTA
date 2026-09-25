#!/usr/bin/env python
# coding: utf-8
"""
ph_pred_rel_mixed_snr.py

Phase + realisation-level SNR prediction in REALISATION MODE.

SNR sampling:
    log10(SNR_r) ~ Uniform(log10(SNR_MIN), log10(SNR_MAX))

This makes the model robust across a broad realisation-SNR range,
e.g. SNR = 10--100.

Input:
    X[r] : (P, L), full PTA realisation

Targets:
    phase[r] : (L,), shared Earth-term GW phase
    snr[r]   : scalar realisation-level SNR

Outputs:
    phase prediction : (B, L, 2) = (cos phi, sin phi)
    SNR prediction   : (B,) = log10(SNR)
"""

import os, math, time, random, platform, csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# Reproducibility
# ==========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {SEED}")


# ==========================================================
# Device
# ==========================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() and platform.system() == "Darwin"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

AMP_ENABLED = device.type == "cuda"
amp_device_type = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")

print("Device:", device, "| AMP:", AMP_ENABLED)

scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)


# ==========================================================
# Config
# ==========================================================
DATA_PATH = "/scratch/subh_phy/work/2Apr/new_method/data/bigdata_3PN/"
NPZ_NAME  = "lr_signals_with_params_E_B_phase_base.npz"

SAVE_DIR = "./checkpoints_phase_tx_realisation_snr_10_100_loguniform"
os.makedirs(SAVE_DIR, exist_ok=True)

VAL_SPLIT  = 0.10
BATCH_SIZE = 256
EPOCHS     = 100

ADD_NOISE = True

# Broad realisation-level SNR range
SNR_MIN = 10.0
SNR_MAX = 100.0
SNR_SAMPLING = "log_uniform"

# For binned validation diagnostics
SNR_BINS = [
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, 50.0),
    (50.0, 100.0),
]

LR   = 2e-4
WD   = 1e-4
CLIP = 1.0

KAPPA    = 8.0
L_SMOOTH = 0.10
L_SPECT  = 0.05
L_SNR    = 0.05

USE_FAST  = True
DS_STRIDE = 4
D_MODEL   = 128
DEPTH     = 4
HEADS     = 4
D_FF      = 512
P_DROP    = 0.1


# ==========================================================
# Helpers
# ==========================================================
def split_realizations(R, val_split=0.15, seed=42):
    rg = np.random.default_rng(seed)
    idx = np.arange(R)
    rg.shuffle(idx)

    n_val = int(round(val_split * R))
    val_r = idx[:n_val]
    trn_r = idx[n_val:]

    return np.sort(trn_r), np.sort(val_r)


def sample_snr_log_uniform(n, snr_min=10.0, snr_max=100.0, seed=0):
    """
    Sample realisation-level SNR over one decade or more:

        log10(SNR_r) ~ Uniform(log10(snr_min), log10(snr_max))
    """
    rg = np.random.default_rng(seed)

    log_lo = math.log10(snr_min)
    log_hi = math.log10(snr_max)

    log_snr = rg.uniform(log_lo, log_hi, size=n)
    snr = 10.0 ** log_snr

    return snr.astype(np.float32)


def add_noise_snr_flat_return_snr(
    X_flat,
    snr_min=10.0,
    snr_max=100.0,
    seed=0,
    sampling="log_uniform",
):
    """
    X_flat: (N, P*L)

    Adds Gaussian noise using one realisation-level SNR per sample.

    Realisation SNR definition:
        SNR_r^2 = ||s_r||^2 / sigma_r^2

    Therefore:
        sigma_r = ||s_r|| / SNR_r
    """
    rg = np.random.default_rng(seed)

    if sampling == "log_uniform":
        snrs = sample_snr_log_uniform(
            X_flat.shape[0],
            snr_min=snr_min,
            snr_max=snr_max,
            seed=seed,
        )
    elif sampling == "uniform":
        snrs = rg.uniform(snr_min, snr_max, size=X_flat.shape[0]).astype(np.float32)
    else:
        raise ValueError(f"Unknown SNR sampling: {sampling}")

    signal_norm = np.sqrt(np.sum(X_flat**2, axis=1)).clip(min=1e-12)
    noise_std = (signal_norm / snrs)[:, None]

    X_noisy = X_flat + rg.normal(0.0, noise_std, size=X_flat.shape)

    return X_noisy.astype(np.float32), snrs.astype(np.float32)


def save_pred_plot(epoch, phi_true, y_pred_unit, out_dir, tag="val0"):
    ang_pred = np.arctan2(y_pred_unit[:, 1], y_pred_unit[:, 0])
    ang_true = phi_true

    ang_pred_u = np.unwrap(ang_pred)
    ang_true_u = np.unwrap(ang_true)

    plt.figure(figsize=(10, 4))
    plt.plot(ang_true_u, label="true phi")
    plt.plot(ang_pred_u, label="predicted phi", alpha=0.85)
    plt.legend()
    plt.xlabel("t index")
    plt.ylabel("phi [rad, unwrapped]")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"phi_epoch_{epoch:04d}_{tag}.png"), dpi=150)
    plt.close()


def save_snr_scatter(epoch, snr_true, snr_pred, out_dir, tag="val_batch0"):
    snr_true = np.asarray(snr_true)
    snr_pred = np.asarray(snr_pred)

    plt.figure(figsize=(5, 5))
    plt.scatter(snr_true, snr_pred, s=12, alpha=0.6)

    lo = min(snr_true.min(), snr_pred.min(), SNR_MIN)
    hi = max(snr_true.max(), snr_pred.max(), SNR_MAX)

    plt.plot([lo, hi], [lo, hi], "k--", lw=1)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("true realisation SNR")
    plt.ylabel("predicted realisation SNR")
    plt.title(f"SNR prediction, epoch {epoch}")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"snr_epoch_{epoch:04d}_{tag}.png"), dpi=150)
    plt.close()


# ==========================================================
# Load data
# ==========================================================
npz_path = os.path.join(DATA_PATH, NPZ_NAME)
print("Loading data from:", npz_path)

data = np.load(npz_path, allow_pickle=True)

X_B     = data["X_B"]       # (P, R, L)
phase_B = data["phase_B"]   # (R, L)

P, R, L = X_B.shape

print("Shapes:")
print("  X_B    :", X_B.shape)
print("  phase_B:", phase_B.shape)

assert phase_B.shape == (R, L)

if USE_FAST:
    assert L % DS_STRIDE == 0, "L must be divisible by DS_STRIDE."

# Center each pulsar time series
X_centered = X_B - X_B.mean(axis=2, keepdims=True)

train_r, val_r = split_realizations(R, VAL_SPLIT, seed=SEED)
print(f"Realisations: total={R}, train={len(train_r)}, val={len(val_r)}")

X_tr = np.transpose(X_centered[:, train_r, :], (1, 0, 2)).astype(np.float32)
X_va = np.transpose(X_centered[:, val_r, :], (1, 0, 2)).astype(np.float32)

X_tr_clean = X_tr.copy()
X_va_clean = X_va.copy()

phi_tr = phase_B[train_r].astype(np.float32)
phi_va = phase_B[val_r].astype(np.float32)


# ==========================================================
# Add log-uniform realisation-level SNR noise
# ==========================================================
if ADD_NOISE:
    Rtr, Pn, Ln = X_tr.shape
    Rva, _, _ = X_va.shape

    X_tr_flat = X_tr.reshape(Rtr, Pn * Ln)
    X_va_flat = X_va.reshape(Rva, Pn * Ln)

    X_tr_flat, snr_tr_np = add_noise_snr_flat_return_snr(
        X_tr_flat,
        snr_min=SNR_MIN,
        snr_max=SNR_MAX,
        seed=123,
        sampling=SNR_SAMPLING,
    )

    X_va_flat, snr_va_np = add_noise_snr_flat_return_snr(
        X_va_flat,
        snr_min=SNR_MIN,
        snr_max=SNR_MAX,
        seed=456,
        sampling=SNR_SAMPLING,
    )

    X_tr = X_tr_flat.reshape(Rtr, Pn, Ln)
    X_va = X_va_flat.reshape(Rva, Pn, Ln)

else:
    snr_tr_np = np.ones(X_tr.shape[0], dtype=np.float32)
    snr_va_np = np.ones(X_va.shape[0], dtype=np.float32)

print("Final samples:")
print("  X_tr:", X_tr.shape)
print("  X_va:", X_va.shape)
print(
    f"SNR train range: [{snr_tr_np.min():.3f}, {snr_tr_np.max():.3f}] | "
    f"SNR val range: [{snr_va_np.min():.3f}, {snr_va_np.max():.3f}]"
)


# ==========================================================
# Standardize inputs
# ==========================================================
X_tr_flat_t = torch.from_numpy(
    np.ascontiguousarray(X_tr.reshape(X_tr.shape[0], -1))
).float()

X_va_flat_t = torch.from_numpy(
    np.ascontiguousarray(X_va.reshape(X_va.shape[0], -1))
).float()

mu_x = X_tr_flat_t.mean(dim=0, keepdim=True)
std_x = X_tr_flat_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

X_tr_t = ((X_tr_flat_t - mu_x) / std_x).reshape(X_tr.shape[0], P, L)
X_va_t = ((X_va_flat_t - mu_x) / std_x).reshape(X_va.shape[0], P, L)

PHI_tr = torch.from_numpy(np.ascontiguousarray(phi_tr)).float()
PHI_va = torch.from_numpy(np.ascontiguousarray(phi_va)).float()

Y_tr = torch.stack([torch.cos(PHI_tr), torch.sin(PHI_tr)], dim=-1)
Y_va = torch.stack([torch.cos(PHI_va), torch.sin(PHI_va)], dim=-1)

snr_tr = torch.from_numpy(snr_tr_np.astype(np.float32))
snr_va = torch.from_numpy(snr_va_np.astype(np.float32))


# ==========================================================
# Save validation data
# ==========================================================
save_dict = {
    "X_va_t": X_va_t.cpu(),
    "X_va_clean": torch.from_numpy(X_va_clean).float(),
    "X_va_noisy": torch.from_numpy(X_va).float(),
    "PHI_va": PHI_va.cpu(),
    "Y_va": Y_va.cpu(),
    "snr_va": snr_va.cpu(),
    "mu_x": mu_x.cpu(),
    "std_x": std_x.cpu(),
    "P": P,
    "L": L,
    "SNR_MIN": SNR_MIN,
    "SNR_MAX": SNR_MAX,
    "SNR_SAMPLING": SNR_SAMPLING,
}

VAL_SAVE_PATH = os.path.join(SAVE_DIR, "val_phase_snr_data_10_100.pt")
torch.save(save_dict, VAL_SAVE_PATH)

print(f"[OK] Saved validation phase+SNR data to: {VAL_SAVE_PATH}")


# ==========================================================
# Dataset
# ==========================================================
class RealisationDataset(Dataset):
    def __init__(self, X_rpl, Y_phase, PHI_phase, snr):
        self.X = X_rpl
        self.Y = Y_phase
        self.PHI = PHI_phase
        self.snr = snr

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.PHI[i], self.snr[i:i+1]


pin = device.type == "cuda"
gen = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    RealisationDataset(X_tr_t, Y_tr, PHI_tr, snr_tr),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin,
    generator=gen,
    num_workers=0,
)

val_loader = DataLoader(
    RealisationDataset(X_va_t, Y_va, PHI_va, snr_va),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin,
    num_workers=0,
)

print("Train batches:", len(train_loader), "| Val batches:", len(val_loader))


# ==========================================================
# Model
# ==========================================================
class SinusoidalPE(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def _normalize_unit(y, eps=1e-7):
    return y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=eps)


class PhaseTransformerRealisation(nn.Module):
    def __init__(self, P, L, d_model=128, depth=4, heads=4, d_ff=512, p_drop=0.1):
        super().__init__()

        self.in_proj = nn.Linear(P, d_model)
        self.pe = SinusoidalPE(L, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.snr_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(-1)

        x = x.permute(0, 2, 1).contiguous()  # (B,L,P)

        z = self.in_proj(x)
        z = self.pe(z)
        z = self.encoder(z)

        y_phase = self.phase_head(z)

        z_pool = z.mean(dim=1)
        snr_log10_pred = self.snr_head(z_pool).squeeze(-1)

        return y_phase, snr_log10_pred


class PhaseTransformerRealisationFast(nn.Module):
    def __init__(
        self,
        P,
        L,
        d_model=128,
        depth=4,
        heads=4,
        d_ff=512,
        p_drop=0.1,
        ds_stride=4,
    ):
        super().__init__()

        assert L % ds_stride == 0

        self.P = P
        self.L = L
        self.S = ds_stride
        self.Ls = L // ds_stride

        self.in_proj = nn.Linear(P, d_model)

        self.ds_conv = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=7,
                stride=ds_stride,
                padding=3,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
        )

        self.post_ds_norm = nn.LayerNorm(d_model)
        self.pe = SinusoidalPE(self.Ls, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_ff,
            dropout=p_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.phase_head_lowrate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.upsampler = nn.Upsample(size=L, mode="linear", align_corners=False)

        self.snr_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(-1)

        x = x.permute(0, 2, 1).contiguous()  # (B,L,P)

        z = self.in_proj(x)
        z = z.transpose(1, 2)
        z = self.ds_conv(z)
        z = z.transpose(1, 2)

        z = self.post_ds_norm(z)
        z = self.pe(z)
        z = self.encoder(z)

        y_low = self.phase_head_lowrate(z)
        y_phase = self.upsampler(y_low.transpose(1, 2)).transpose(1, 2)

        z_pool = z.mean(dim=1)
        snr_log10_pred = self.snr_head(z_pool).squeeze(-1)

        return y_phase, snr_log10_pred


model = (
    PhaseTransformerRealisationFast(
        P,
        L,
        d_model=D_MODEL,
        depth=DEPTH,
        heads=HEADS,
        d_ff=D_FF,
        p_drop=P_DROP,
        ds_stride=DS_STRIDE,
    )
    if USE_FAST
    else PhaseTransformerRealisation(
        P,
        L,
        d_model=D_MODEL,
        depth=DEPTH,
        heads=HEADS,
        d_ff=D_FF,
        p_drop=P_DROP,
    )
).to(device)

print("Model params: %.3fM" % (sum(p.numel() for p in model.parameters()) / 1e6))


# ==========================================================
# Losses and metrics
# ==========================================================
def von_mises_nll(y_pred, y_true, kappa=KAPPA):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)
    cosd = torch.sum(yp * yt, dim=-1)
    return torch.mean(-kappa * cosd)


def circular_smoothness_loss(y):
    y = _normalize_unit(y)

    a = torch.atan2(y[..., 1], y[..., 0])
    d = a[:, 1:] - a[:, :-1]
    d = (d + math.pi) % (2 * math.pi) - math.pi

    return torch.mean(d**2)


def spectral_unit_loss(y_pred, y_true):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)

    zp = torch.complex(yp[..., 0], yp[..., 1])
    zt = torch.complex(yt[..., 0], yt[..., 1])

    with torch.autocast(device_type=amp_device_type, enabled=False):
        Pp = torch.fft.fft(zp, dim=-1)
        Pt = torch.fft.fft(zt, dim=-1)

        Pm = Pp.abs()
        Tm = Pt.abs()

        Pm = Pm / (Pm.norm(dim=-1, keepdim=True) + 1e-8)
        Tm = Tm / (Tm.norm(dim=-1, keepdim=True) + 1e-8)

        return F.mse_loss(Pm, Tm)


def total_loss(y_pred, y_true, snr_log10_pred=None, snr_true=None):
    lv = von_mises_nll(y_pred, y_true)
    ls = circular_smoothness_loss(y_pred)
    lp = spectral_unit_loss(y_pred, y_true)

    phase_loss = lv + L_SMOOTH * ls + L_SPECT * lp

    if (snr_log10_pred is not None) and (snr_true is not None):
        snr_true_log10 = torch.log10(torch.clamp(snr_true.squeeze(-1), min=1e-3))
        snr_loss = F.mse_loss(snr_log10_pred, snr_true_log10)
    else:
        snr_loss = torch.zeros((), device=y_pred.device)

    total = phase_loss + L_SNR * snr_loss

    return total, dict(
        total=total.item(),
        phase=phase_loss.item(),
        snr=snr_loss.item(),
        von_mises=lv.item(),
        smooth=ls.item(),
        spectral=lp.item(),
    )


@torch.no_grad()
def angular_mae_deg_per_sample(y_pred, y_true):
    yp = _normalize_unit(y_pred)
    yt = _normalize_unit(y_true)

    ap = torch.atan2(yp[..., 1], yp[..., 0])
    at = torch.atan2(yt[..., 1], yt[..., 0])

    d = (ap - at + math.pi) % (2 * math.pi) - math.pi

    return d.abs().mean(dim=1).mul(180.0 / math.pi)


@torch.no_grad()
def angular_mae_deg(y_pred, y_true):
    return angular_mae_deg_per_sample(y_pred, y_true).mean().item()


@torch.no_grad()
def snr_mae_metrics(snr_log10_pred, snr_true):
    snr_pred = torch.pow(10.0, snr_log10_pred)
    snr_true = snr_true.squeeze(-1)

    mae = torch.mean(torch.abs(snr_pred - snr_true)).item()
    rmse = torch.sqrt(torch.mean((snr_pred - snr_true) ** 2)).item()

    return mae, rmse


# ==========================================================
# Optimizer
# ==========================================================
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)


# ==========================================================
# Output files
# ==========================================================
best_path = os.path.join(
    SAVE_DIR,
    "phase_predictor_best_fast.pt" if USE_FAST else "phase_predictor_best_slow.pt"
)

last_path = os.path.join(
    SAVE_DIR,
    "phase_predictor_last_fast.pt" if USE_FAST else "phase_predictor_last_slow.pt"
)

csv_path = os.path.join(SAVE_DIR, "loss_log.csv")
png_path = os.path.join(SAVE_DIR, "loss_curves.png")
component_png_path = os.path.join(SAVE_DIR, "loss_components.png")
bin_csv_path = os.path.join(SAVE_DIR, "validation_snr_bins.csv")

PLOT_DIR = os.path.join(SAVE_DIR, "epoch_phase_plots")
SNR_PLOT_DIR = os.path.join(SAVE_DIR, "epoch_snr_plots")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SNR_PLOT_DIR, exist_ok=True)

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_total", "train_phase", "train_snr",
            "train_vm", "train_sm", "train_sp",
            "val_total", "val_phase", "val_snr",
            "val_vm", "val_sm", "val_sp",
            "val_angMAE_deg",
            "val_snr_MAE", "val_snr_RMSE",
            "lr", "epoch_time_s",
        ])


epochs_hist = []
train_hist = []
val_hist = []
best_val = float("inf")


def save_loss_plot():
    if len(epochs_hist) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_hist, train_hist, label="train", marker="o")
    plt.plot(epochs_hist, val_hist, label="val", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def save_component_loss_plot():
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_total"], label="train")
    axes[0].plot(df["epoch"], df["val_total"], label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_phase"], label="train")
    axes[1].plot(df["epoch"], df["val_phase"], label="val")
    axes[1].set_title("Phase loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["epoch"], df["train_snr"], label="train")
    axes[2].plot(df["epoch"], df["val_snr"], label="val")
    axes[2].set_title("SNR loss")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.savefig(component_png_path, dpi=150)
    plt.close()


def run_epoch(loader, train=True):
    model.train(train)

    n = 0

    logs = {
        "total": 0.0,
        "phase": 0.0,
        "snr": 0.0,
        "von_mises": 0.0,
        "smooth": 0.0,
        "spectral": 0.0,
    }

    for xb, yb, phib, snb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        snb = snb.to(device)

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=amp_device_type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            y_pred, snr_log10_pred = model(xb)
            loss, parts = total_loss(y_pred, yb, snr_log10_pred, snb)

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()

        bs = xb.size(0)
        n += bs

        for k in logs:
            logs[k] += parts[k] * bs

    for k in logs:
        logs[k] /= max(1, n)

    return logs["total"], logs


# ==========================================================
# Validation collection and binned diagnostics
# ==========================================================
@torch.no_grad()
def collect_validation_predictions():
    model.eval()

    all_phase_err = []
    all_snr_true = []
    all_snr_pred = []

    example = None

    for xb, yb, phib, snb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        snb_dev = snb.to(device)

        y_pred, snr_log10_pred = model(xb)

        phase_err = angular_mae_deg_per_sample(y_pred, yb)
        snr_pred = torch.pow(10.0, snr_log10_pred)

        all_phase_err.append(phase_err.cpu())
        all_snr_true.append(snb.squeeze(-1).cpu())
        all_snr_pred.append(snr_pred.cpu())

        if example is None:
            yp0 = _normalize_unit(y_pred[0]).detach().cpu().numpy()
            example = (
                xb[0].detach().cpu().numpy(),
                phib[0].detach().cpu().numpy(),
                yp0,
                snb[0].item(),
                snr_pred[0].item(),
            )

    phase_err = torch.cat(all_phase_err).numpy()
    snr_true = torch.cat(all_snr_true).numpy()
    snr_pred = torch.cat(all_snr_pred).numpy()

    return phase_err, snr_true, snr_pred, example


def compute_binned_metrics(phase_err, snr_true, snr_pred, bins=SNR_BINS):
    rows = []

    for lo, hi in bins:
        mask = (snr_true >= lo) & (snr_true < hi)

        if not np.any(mask):
            rows.append({
                "snr_lo": lo,
                "snr_hi": hi,
                "n": 0,
                "phase_mae_deg": np.nan,
                "snr_mae": np.nan,
                "snr_rmse": np.nan,
                "snr_bias": np.nan,
            })
            continue

        err_snr = snr_pred[mask] - snr_true[mask]

        rows.append({
            "snr_lo": lo,
            "snr_hi": hi,
            "n": int(mask.sum()),
            "phase_mae_deg": float(np.mean(phase_err[mask])),
            "snr_mae": float(np.mean(np.abs(err_snr))),
            "snr_rmse": float(np.sqrt(np.mean(err_snr**2))),
            "snr_bias": float(np.mean(err_snr)),
        })

    return rows


def save_binned_metrics_csv(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[OK] Saved binned validation metrics: {path}")
    print(df)


def save_binned_metrics_plot(rows, save_path):
    df = pd.DataFrame(rows)
    df = df[df["n"] > 0].copy()

    if len(df) == 0:
        return

    labels = [f"{r.snr_lo:.0f}-{r.snr_hi:.0f}" for _, r in df.iterrows()]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

    ax1.plot(x, df["phase_mae_deg"], marker="o", label="phase MAE [deg]")
    ax1.set_ylabel("phase MAE [deg]")
    ax1.set_xlabel("true SNR bin")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, df["snr_mae"], marker="s", linestyle="--", label="SNR MAE")
    ax2.set_ylabel("SNR MAE")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Validation performance by realisation-SNR bin")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

    print(f"[OK] Saved binned metrics plot: {save_path}")


# ==========================================================
# Four-panel paper-style validation plot
# ==========================================================
@torch.no_grad()
def make_phase_snr_summary_plot(save_name="phase_alignment_snr_10_100_loguniform.png"):
    phase_err, snr_true, snr_pred, example = collect_validation_predictions()

    x_ex_std, phi_true, y_pred_unit, snr_t_ex, snr_p_ex = example

    phi_pred = np.arctan2(y_pred_unit[:, 1], y_pred_unit[:, 0])

    phi_true_u = np.unwrap(phi_true)
    phi_pred_u = np.unwrap(phi_pred)

    dphi_wrapped = (phi_pred - phi_true + np.pi) % (2 * np.pi) - np.pi
    dphi_ex_deg = np.mean(np.abs(dphi_wrapped)) * 180.0 / np.pi

    # use first validation realisation and first pulsar for clean/noisy signal panel
    x_clean_ex = X_va_clean[0, 0]
    x_noisy_ex = X_va[0, 0]

    t = np.linspace(0, 12, L)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # 1. Phase-error histogram
    ax = axes[0]
    ax.hist(phase_err, bins=50, density=True, alpha=0.65)
    ax.axvline(
        np.mean(phase_err),
        color="r",
        ls="--",
        lw=1,
        label=rf"mean = {np.mean(phase_err):.2f}$^\circ$",
    )
    ax.axvline(
        np.median(phase_err),
        color="k",
        ls=":",
        lw=1,
        label=rf"median = {np.median(phase_err):.2f}$^\circ$",
    )
    ax.set_title("Phase alignment over validation set")
    ax.set_xlabel(r"$\langle|\Delta\phi|\rangle$ [deg]")
    ax.set_ylabel("PDF")
    ax.legend(fontsize=8)

    # 2. SNR prediction
    ax = axes[1]
    ax.scatter(snr_true, snr_pred, s=4, alpha=0.18, label="predicted SNR")
    ax.plot([SNR_MIN, SNR_MAX], [SNR_MIN, SNR_MAX], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("SNR prediction")
    ax.set_xlabel("true SNR")
    ax.set_ylabel("predicted SNR")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # 3. Clean vs noisy signal
    ax = axes[2]
    ax.plot(t, x_clean_ex, lw=1.2, alpha=0.7, label="clean signal")
    ax.scatter(t, x_noisy_ex, s=5, alpha=0.45, label="noisy signal")
    ax.set_title("Clean vs noisy signal")
    ax.set_xlabel("time [yr]")
    ax.set_ylabel("signal")
    ax.legend(fontsize=8)

    # 4. Phase recovery
    ax = axes[3]
    ax.plot(t, phi_true_u, lw=1.5, label=r"true $\phi(t)$")
    ax.plot(t, phi_pred_u, lw=1.5, alpha=0.85, label=r"predicted $\phi(t)$")
    ax.set_title(
        rf"$\Delta\phi={dphi_ex_deg:.2f}^\circ$, "
        rf"SNR={snr_t_ex:.1f}$\rightarrow${snr_p_ex:.1f}"
    )
    ax.set_xlabel("time [yr]")
    ax.set_ylabel(r"$\phi(t)$")
    ax.legend(fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"[OK] Saved summary plot: {out_path}")


# ==========================================================
# Training loop
# ==========================================================
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    tr, trp = run_epoch(train_loader, train=True)
    va, vap = run_epoch(val_loader, train=False)

    sched.step()
    lr = sched.get_last_lr()[0]

    with torch.no_grad():
        xb, yb, phib, snb = next(iter(val_loader))

        xb = xb.to(device)
        yb = yb.to(device)
        snb = snb.to(device)

        y_pred, snr_log10_pred = model(xb)

        ang_mae = angular_mae_deg(y_pred, yb)
        snr_mae, snr_rmse = snr_mae_metrics(snr_log10_pred, snb)

        y0 = _normalize_unit(y_pred[0]).detach().cpu().numpy()
        phi0 = phib[0].numpy()

        save_pred_plot(epoch, phi0, y0, PLOT_DIR, tag="val0")

        snr_pred_np = torch.pow(10.0, snr_log10_pred).detach().cpu().numpy()
        snr_true_np = snb.squeeze(-1).detach().cpu().numpy()

        save_snr_scatter(epoch, snr_true_np, snr_pred_np, SNR_PLOT_DIR, tag="val_batch0")

    dt = time.time() - t0

    print(
        f"[{epoch:03d}/{EPOCHS:03d}] "
        f"train {tr:.6f} | val {va:.6f} | "
        f"angMAE {ang_mae:.3f} deg | "
        f"SNR_MAE {snr_mae:.3f} | SNR_RMSE {snr_rmse:.3f} | "
        f"lr {lr:.3e} | {dt:.1f}s"
    )

    epochs_hist.append(epoch)
    train_hist.append(tr)
    val_hist.append(va)

    save_loss_plot()

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            trp["total"], trp["phase"], trp["snr"],
            trp["von_mises"], trp["smooth"], trp["spectral"],
            vap["total"], vap["phase"], vap["snr"],
            vap["von_mises"], vap["smooth"], vap["spectral"],
            ang_mae,
            snr_mae,
            snr_rmse,
            lr,
            dt,
        ])

    save_component_loss_plot()

    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "mu_x": mu_x.cpu(),
        "std_x": std_x.cpu(),
        "P": P,
        "L": L,
        "USE_FAST": USE_FAST,
        "DS_STRIDE": DS_STRIDE,
        "D_MODEL": D_MODEL,
        "DEPTH": DEPTH,
        "HEADS": HEADS,
        "D_FF": D_FF,
        "P_DROP": P_DROP,
        "ADD_NOISE": ADD_NOISE,
        "SNR_MIN": SNR_MIN,
        "SNR_MAX": SNR_MAX,
        "SNR_SAMPLING": SNR_SAMPLING,
        "L_SNR": L_SNR,
        "predicts_snr": True,
        "snr_target": "realisation_log10_snr",
    }

    if va < best_val:
        best_val = va
        ckpt["best_val"] = best_val
        torch.save(ckpt, best_path)

    ckpt["best_val"] = best_val
    torch.save(ckpt, last_path)


print("Training complete.")
print("Best checkpoint:", best_path)
print("Last checkpoint:", last_path)
print("Loss curve PNG:", png_path)
print("Loss components PNG:", component_png_path)
print("Per-epoch phase plots:", PLOT_DIR)
print("Per-epoch SNR plots:", SNR_PLOT_DIR)


# ==========================================================
# Final validation summary
# ==========================================================
@torch.no_grad()
def final_validation_summary():
    phase_err, snr_true, snr_pred, _ = collect_validation_predictions()

    snr_mae = np.mean(np.abs(snr_pred - snr_true))
    snr_rmse = np.sqrt(np.mean((snr_pred - snr_true) ** 2))

    print("\nFinal validation summary")
    print("------------------------")
    print(f"Phase MAE [deg] : {np.mean(phase_err):.4f}")
    print(f"SNR MAE         : {snr_mae:.4f}")
    print(f"SNR RMSE        : {snr_rmse:.4f}")

    rows = compute_binned_metrics(phase_err, snr_true, snr_pred, SNR_BINS)

    save_binned_metrics_csv(rows, bin_csv_path)
    save_binned_metrics_plot(
        rows,
        os.path.join(SAVE_DIR, "validation_snr_bin_metrics.png"),
    )

    plt.figure(figsize=(5, 5))
    plt.scatter(snr_true, snr_pred, s=8, alpha=0.30)
    plt.plot([SNR_MIN, SNR_MAX], [SNR_MIN, SNR_MAX], "k--", lw=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("true realisation SNR")
    plt.ylabel("predicted realisation SNR")
    plt.title("Final validation SNR prediction")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "final_snr_scatter_10_100.png"), dpi=180)
    plt.close()


final_validation_summary()
make_phase_snr_summary_plot()



OUT_NPZ = os.path.join(OUTDIR, "phase_histogram_data.npz")

np.savez(
    OUT_NPZ,
    phase_mae_per_sample=phase_mae_per_sample,   # (Nval,)
    phase_err_full=phase_err_deg,                # (Nval, L)
    snr_true=snr_true_all,                      # (Nval,)
    snr_pred=snr_pred_all,                      # (Nval,)
    mean_phase=mean_phase,
    median_phase=median_phase,
)

print("Saved histogram NPZ:")
print(OUT_NPZ)
