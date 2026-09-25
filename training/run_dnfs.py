#!/usr/bin/env python
# coding: utf-8

import os, sys, random, platform, csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
# PATHS
# ==========================================================
PKG_PATH = "/../packages/"
sys.path.insert(0, PKG_PATH)

from phase_predictor import PhaseProvider
import hierarchical_dnf as mdl
from hierarchical_dnf import PosteriorNet

# ==========================================================
# GLOBAL REPRODUCIBILITY
# ==========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {SEED}")

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() and platform.system() == "Darwin"
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# IMPORTANT FIX: disable AMP for flow stability
AMP_ENABLED = False

print("Device:", device, "| AMP:", AMP_ENABLED)

# ==========================================================
# CONFIG
# ==========================================================
DATA_PATH  = "/scratch/projects/CFP03/CFP03-CF-051/projects/SBI/2Apr/new_method/realisation_paper/data/mid"
NPZ_NAME   = "lr_signals_3PN_E_B_phase.npz"

VAL_SPLIT  = 0.10
BATCH_SIZE = 128
EPOCHS     = 150

RESUME = False  # first self-attention run must start from scratch

LR   = 1e-4
WD   = 1e-4
CLIP = 0.5

# early stopping
EARLY_STOP = True
PATIENCE   = 15
MIN_DELTA  = 1e-3

# LR scheduler
USE_LR_SCHEDULER = True
LR_PATIENCE = 4
LR_FACTOR   = 0.5
LR_MIN      = 1e-6

target_names = ["log10_n","e0", "log10_M", "log10_A"]

phase_target_names = ["log10_n","e0", "log10_M"]

USE_TRUE_PHASE     = False
USE_PHASE_PROVIDER = True
USE_PHASE          = bool(USE_TRUE_PHASE or USE_PHASE_PROVIDER)

ADD_NOISE = True
SNR_LO, SNR_HI = 20, 30

SAVE_DIR = "dnfs_pred_phase_masked_hierarchical_self_attention"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, "best_posterior_flow_dnfs_pred_phase_masked_hierarchical_self_attention.pt")
LAST_PATH = os.path.join(SAVE_DIR, "last_posterior_flow_dnfs_pred_phase_masked_hierarchical_self_attention.pt")
CSV_PATH  = os.path.join(SAVE_DIR, "loss_log_dnfs_pred_phase_self_attention.csv")

# ==========================================================
# HELPERS
# ==========================================================
def split_realizations(R: int, val_split: float = 0.10, seed: int = 42):
    rg = np.random.default_rng(seed)
    idx = np.arange(R)
    rg.shuffle(idx)
    n_val = int(round(val_split * R))
    val_r = idx[:n_val]
    trn_r = idx[n_val:]
    return np.sort(trn_r), np.sort(val_r)

def add_noise_snr_flat(X: np.ndarray, snr_lo: int, snr_hi: int, seed: int):
    rg = np.random.default_rng(seed)
    snrs = rg.integers(snr_lo, snr_hi + 1, size=X.shape[0])
    s_x = np.sqrt(np.sum(X**2, axis=1)).clip(min=1e-12)
    sigmas = (s_x / snrs).astype(np.float64)
    noise = rg.normal(0.0, sigmas[:, None], size=X.shape)
    return X + noise, snrs.astype(np.int64), sigmas

def standardize_realisation_X(X_tr: np.ndarray, X_va: np.ndarray):
    Rtr, P, L = X_tr.shape
    Rva, _, _ = X_va.shape

    X_tr_flat = torch.as_tensor(X_tr.reshape(Rtr, -1), dtype=torch.float32)
    X_va_flat = torch.as_tensor(X_va.reshape(Rva, -1), dtype=torch.float32)

    X_mean = X_tr_flat.mean(dim=0, keepdim=True)
    X_std  = X_tr_flat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)

    X_tr_t = ((X_tr_flat - X_mean) / X_std).reshape(Rtr, P, L)
    X_va_t = ((X_va_flat - X_mean) / X_std).reshape(Rva, P, L)

    return X_tr_t, X_va_t, X_mean, X_std

def standardize_y(y_tr: np.ndarray, y_va: np.ndarray):
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
    y_va_t = torch.as_tensor(y_va, dtype=torch.float32)

    y_mean = y_tr_t.mean(dim=0, keepdim=True)
    y_std  = y_tr_t.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)

    y_tr_t = (y_tr_t - y_mean) / y_std
    y_va_t = (y_va_t - y_mean) / y_std

    return y_tr_t, y_va_t, y_mean, y_std

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            xb, yb = batch
            return xb, None, yb
        if len(batch) == 3:
            xb, phib, yb = batch
            return xb, phib, yb
    raise ValueError("Batch must be (xb, yb) or (xb, phib, yb)")

def prepare_batch(batch, device):
    xb, phib, yb = _unpack_batch(batch)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    if phib is not None:
        phib = phib.to(device, non_blocking=True)
    return xb, phib, yb

def _read_scalar_param(module, name):
    if hasattr(module, name):
        x = getattr(module, name)
        if torch.is_tensor(x):
            return x.detach().cpu().item()
        if isinstance(x, (float, int)):
            return float(x)
    return None

def get_phase_weights(model):
    try:
        base_temporal  = model.cond_base.net.temporal
        phase_temporal = model.cond_phase.net.temporal

        base_w_pos     = _read_scalar_param(base_temporal, "w_pos")
        phase_w_pos    = _read_scalar_param(phase_temporal, "w_pos")
        phase_w_phase  = _read_scalar_param(phase_temporal, "w_phase")

        return base_w_pos, phase_w_pos, phase_w_phase
    except Exception:
        return None, None, None

def get_lr(opt):
    return opt.param_groups[0]["lr"]

def make_checkpoint(
    ep, model, opt, scaler, scheduler, best_val, epochs_no_improve,
    target_names, phase_target_names, phase_target_idx,
    THETA_DIM, P, L, X_mean, X_std, y_mean, y_std,
    base_w_pos, phase_w_pos, phase_w_phase
):
    ckpt = {
        "epoch": ep,
        "best_val": best_val,
        "epochs_no_improve": epochs_no_improve,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),

        "target_names": target_names,
        "phase_target_names": phase_target_names,
        "phase_target_idx": phase_target_idx,
        "THETA_DIM": THETA_DIM,
        "attention_type": "self_attention",
        "P": P,
        "L": L,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "base_w_pos": base_w_pos,
        "phase_w_pos": phase_w_pos,
        "phase_w_phase": phase_w_phase,
    }

    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()

    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()

    return ckpt

# ==========================================================
# SECTION 1 — LOAD NPZ
# ==========================================================
npz_path = os.path.join(DATA_PATH, NPZ_NAME)
print("Loading:", npz_path)
data = np.load(npz_path, allow_pickle=True)

# For mid file, use E-mode signal
X_B      = data["X_B"]
Y_by_psr = data["Y_by_pulsar"]
phase_B  = data["phase_B"]
param_cols = list(data["param_cols"])

P, R, L = X_B.shape
print("Shapes:")
print("  X_B      :", X_B.shape)
print("  Y_by_psr :", Y_by_psr.shape)
print("  phase_B  :", phase_B.shape)
print("  P,R,L    :", P, R, L)

# ==========================================================
# SECTION 2 — ENSURE log10_Mc EXISTS
# ==========================================================
if "log10_Mc" not in param_cols:
    if ("log10_M" not in param_cols) or ("q" not in param_cols):
        raise KeyError("Need 'log10_M' and 'q' in param_cols to derive log10_Mc.")

    iM = param_cols.index("log10_M")
    iq = param_cols.index("q")

    log10_M = Y_by_psr[:, :, iM]
    q       = Y_by_psr[:, :, iq]
    eta     = q / (1.0 + q)**2
    Mc      = (10.0**log10_M) * (eta**(3.0/5.0))
    log10_Mc = np.log10(Mc)

    Y_by_psr = np.concatenate([Y_by_psr, log10_Mc[..., None]], axis=2)
    param_cols.append("log10_Mc")
    print("Derived and appended log10_Mc. New nparam =", len(param_cols))

name_to_idx = {n: i for i, n in enumerate(param_cols)}

for n in target_names:
    if n not in name_to_idx:
        raise KeyError(f"target '{n}' not in param_cols={param_cols}")

for n in phase_target_names:
    if n not in target_names:
        raise KeyError(f"phase target '{n}' not in target_names={target_names}")

tidx = [name_to_idx[n] for n in target_names]
THETA_DIM = len(tidx)
mdl.THETA_DIM = THETA_DIM

phase_target_idx = [target_names.index(n) for n in phase_target_names]

print("THETA_DIM =", THETA_DIM)
print("Masked phase targets:", list(zip(phase_target_names, phase_target_idx)))

# ==========================================================
# SECTION 3 — CENTER X PER PULSAR
# ==========================================================
X_centered = X_B - X_B.mean(axis=2, keepdims=True)

# ==========================================================
# SECTION 4 — ADD NOISE TO ENTIRE DATASET FIRST
# ==========================================================
X_all_clean = np.transpose(X_centered, (1, 0, 2))  # (R,P,L)

if ADD_NOISE:
    print(f"Adding noise to ENTIRE dataset with SNR in [{SNR_LO}, {SNR_HI}] inclusive.")
    X_all_flat = X_all_clean.reshape(R, P * L)
    X_all_flat_noisy, snr_all, sigma_all = add_noise_snr_flat(
        X_all_flat, SNR_LO, SNR_HI, seed=SEED + 12345
    )
    X_all = X_all_flat_noisy.reshape(R, P, L)
else:
    X_all = X_all_clean.copy()
    snr_all = np.full((R,), -1, dtype=np.int64)
    sigma_all = np.zeros((R,), dtype=np.float64)

log10_sigma_all = np.log10(np.clip(sigma_all, 1e-300, None)).astype(np.float64)

# ==========================================================
# SECTION 5 — SPLIT
# ==========================================================
train_r, val_r = split_realizations(R, val_split=VAL_SPLIT, seed=SEED)
print(f"Split realizations: total={R} | train={len(train_r)} | val={len(val_r)}")

X_tr = X_all[train_r]
X_va = X_all[val_r]

y_tr = Y_by_psr[0, train_r, :][:, tidx]
y_va = Y_by_psr[0, val_r,   :][:, tidx]

if USE_TRUE_PHASE:
    phi_tr = np.repeat(phase_B[train_r][:, None, :], P, axis=1)
    phi_va = np.repeat(phase_B[val_r][:, None, :],   P, axis=1)
else:
    phi_tr = phi_va = None

print("Final arrays:")
print("  X_tr:", X_tr.shape, "X_va:", X_va.shape)
print("  y_tr:", y_tr.shape, "y_va:", y_va.shape)

# ==========================================================
# SECTION 6 — STANDARDIZATION
# ==========================================================
X_tr_t, X_va_t, X_mean, X_std = standardize_realisation_X(X_tr, X_va)
y_tr_t, y_va_t, y_mean, y_std = standardize_y(y_tr, y_va)

if USE_TRUE_PHASE:
    phi_tr_t = torch.as_tensor(phi_tr, dtype=torch.float32)
    phi_va_t = torch.as_tensor(phi_va, dtype=torch.float32)

print("Torch tensors:")
print("  X_tr_t:", tuple(X_tr_t.shape), "X_va_t:", tuple(X_va_t.shape))
print("  y_tr_t:", tuple(y_tr_t.shape), "y_va_t:", tuple(y_va_t.shape))
print("  X_mean/std:", tuple(X_mean.shape), tuple(X_std.shape))

# ==========================================================
# SECTION 7 — DATALOADERS
# ==========================================================
pin = (device.type == "cuda")
loader_gen = torch.Generator().manual_seed(SEED)

if USE_TRUE_PHASE:
    train_ds = TensorDataset(X_tr_t, phi_tr_t, y_tr_t)
    val_ds   = TensorDataset(X_va_t, phi_va_t, y_va_t)
else:
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds   = TensorDataset(X_va_t, y_va_t)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin,
    generator=loader_gen,
    num_workers=0,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin,
    num_workers=0,
)

print("Batches:", "train =", len(train_loader), "| val =", len(val_loader))

# ==========================================================
# SECTION 8 — PHASE PROVIDER
# ==========================================================
phase_provider = None

if USE_PHASE_PROVIDER:
    PHASE_CKPT = os.path.join(PKG_PATH, "phase_predictor_best_fast.pt")
    phase_provider = PhaseProvider(
        phase_ckpt_path=PHASE_CKPT,
        device=device,
        base_len=L,
    )
    print("PhaseProvider ready:", PHASE_CKPT)
else:
    print("PhaseProvider disabled.")

# ==========================================================
# SECTION 9 — BUILD MODEL
# ==========================================================
model = PosteriorNet(
    seq_len=L,
    n_pulsars=P,
    use_phase=USE_PHASE,
    phase_provider=phase_provider,
    x_mean=X_mean,
    x_std=X_std,
    target_names=target_names,
    phase_target_names=phase_target_names,
).to(device)

# Verify that the imported conditioner really uses standard self-attention.
parameter_names = [name for name, _ in model.named_parameters()]
has_self_attention = any("self_attn.in_proj_weight" in name for name in parameter_names)
has_external_memory = any(
    name.endswith(".Mk") or name.endswith(".Mv") or ".Mk" in name or ".Mv" in name
    for name in parameter_names
)

if not has_self_attention:
    raise RuntimeError(
        "No nn.MultiheadAttention parameters were found. "
        "Check that model_dnfs_masked_hierarchical_self_attention imports "
        "HierarchicalPTAEncoder from self_attention_model_hy_clean."
    )

if has_external_memory:
    raise RuntimeError(
        "External-attention memory parameters Mk/Mv were found. "
        "The wrong encoder module is being imported."
    )

print("Attention backend verified: standard multi-head self-attention.")

print("\nTrainable PE weights:")
for name, p in model.named_parameters():
    if "w_pos" in name or "w_phase" in name:
        print(name, p.requires_grad, p.detach().cpu().item())
print()

base_w_pos, phase_w_pos, phase_w_phase = get_phase_weights(model)
print("Initial logged PE weights:")
print("  base_w_pos    =", base_w_pos)
print("  phase_w_pos   =", phase_w_pos)
print("  phase_w_phase =", phase_w_phase)
print()

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

# AMP disabled, but keep scaler variable for checkpoint compatibility
scaler = None

scheduler = None
if USE_LR_SCHEDULER:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        threshold=MIN_DELTA,
        threshold_mode="abs",
        min_lr=LR_MIN,
    )

best_val = float("inf")
START_EPOCH = 1
epochs_no_improve = 0

# ==========================================================
# RESUME LOGIC
# ==========================================================
if RESUME:
    resume_path = LAST_PATH if os.path.exists(LAST_PATH) else SAVE_PATH

    if os.path.exists(resume_path):
        print(f"\nRESUME=True: loading checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        ckpt_attention = ckpt.get("attention_type")
        if ckpt_attention not in {None, "self_attention"}:
            raise RuntimeError(
                f"Checkpoint attention_type={ckpt_attention!r}; "
                "expected 'self_attention'."
            )

        # Older checkpoints without metadata are accepted only when their
        # parameter keys match the self-attention architecture.
        ckpt_keys = ckpt["model_state"].keys()
        if any(".Mk" in key or ".Mv" in key for key in ckpt_keys):
            raise RuntimeError(
                "This checkpoint contains external-attention Mk/Mv parameters. "
                "Use a new self-attention checkpoint directory and retrain."
            )

        model.load_state_dict(ckpt["model_state"], strict=True)

        if "opt_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt_state"])
                print("Loaded optimizer state.")
            except Exception as e:
                print("Warning: could not load optimizer state:", repr(e))
                print("Optimizer starts fresh.")
        else:
            print("Warning: optimizer state not found; optimizer starts fresh.")

        if scheduler is not None and "scheduler_state" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
                print("Loaded scheduler state.")
            except Exception as e:
                print("Warning: could not load scheduler state:", repr(e))

        START_EPOCH = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

        print(f"Resume starts from epoch {START_EPOCH}")
        print(f"Previous best_val = {best_val:.6f}")
        print(f"Previous epochs_no_improve = {epochs_no_improve}")
        print(f"Current LR = {get_lr(opt):.3e}\n")
    else:
        print("\nRESUME=True but no checkpoint found. Starting from scratch.\n")
else:
    print("\nRESUME=False: starting from scratch.\n")

# ==========================================================
# TRAIN/EVAL
# ==========================================================
header = (
    ["epoch", "train_nll", "val_nll", "lr",
     "base_w_pos", "phase_w_pos", "phase_w_phase",
     "epochs_no_improve"]
    + [f"val_z2_{t}" for t in target_names]
)

if (not RESUME) or (not os.path.exists(CSV_PATH)):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

def train_epoch(model, loader):
    model.train()
    total, nobs = 0.0, 0
    skipped = 0

    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)
        opt.zero_grad(set_to_none=True)

        nll = -model.log_prob(yb, xb, phib).mean()

        if not torch.isfinite(nll):
            print("WARNING: non-finite train loss. Skipping batch.")
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue

        nll.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        if not torch.isfinite(grad_norm):
            print("WARNING: non-finite gradient. Skipping optimizer step.")
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue

        opt.step()

        bs = xb.size(0)
        total += float(nll.item()) * bs
        nobs += bs

    if skipped > 0:
        print(f"Skipped {skipped} unstable train batches.")

    return total / max(nobs, 1)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, nobs = 0.0, 0

    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)
        nll = -model.log_prob(yb, xb, phib).mean()

        if not torch.isfinite(nll):
            print("WARNING: non-finite validation loss detected.")
            return float("inf")

        bs = xb.size(0)
        total += float(nll.item()) * bs
        nobs += bs

    return total / max(nobs, 1)

@torch.no_grad()
def per_target_z2(model, loader):
    model.eval()
    acc = torch.zeros(len(target_names))
    nb = 0

    for batch in loader:
        xb, phib, yb = prepare_batch(batch, device)
        z, _ = model.fwd_to_z(yb, xb, phib if model.use_phase else None)

        z2 = z ** 2

        if not torch.isfinite(z2).all():
            return torch.full((len(target_names),), float("inf"))

        acc += z2.mean(dim=0).detach().cpu()
        nb += 1

    return acc / max(nb, 1)

# ==========================================================
# LIVE PLOTS
# ==========================================================
def _save_live_plots(
    train_losses,
    val_losses,
    z2_hist,
    base_w_pos,
    phase_w_pos,
    phase_w_phase,
    save_dir,
    epoch,
    snap_every=1,
    show_every=0,
):
    fig = plt.figure(figsize=(7.4, 4.6))
    plt.plot(train_losses, label="train_nll")
    plt.plot(val_losses, label="val_nll")
    plt.xlabel("epoch index in this run")
    plt.ylabel("NLL")
    plt.grid(ls="--", alpha=0.4)
    plt.legend()

    if base_w_pos is not None and phase_w_pos is not None and phase_w_phase is not None:
        plt.title(
            f"Epoch {epoch:04d} | LR={get_lr(opt):.2e} | "
            f"base_w_pos={base_w_pos:.5f} | phase_w_pos={phase_w_pos:.5f} | "
            f"phase_w_phase={phase_w_phase:.5f}",
            fontsize=8,
        )
    else:
        plt.title(f"Epoch {epoch:04d} | LR={get_lr(opt):.2e}", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "live_loss.png"), dpi=160)

    if snap_every and (epoch % snap_every == 0):
        fig.savefig(os.path.join(save_dir, f"live_loss_ep{epoch:04d}.png"), dpi=160)

    if show_every and (epoch % show_every == 0):
        plt.show()

    plt.close(fig)

    if z2_hist is not None and len(z2_hist) > 0:
        z2_arr = np.stack(z2_hist, axis=0)

        fig2 = plt.figure(figsize=(7.5, 4.2))
        for j, name in enumerate(target_names):
            plt.plot(z2_arr[:, j], label=f"z2_{name}")

        plt.xlabel("epoch index in this run")
        plt.ylabel("E[z^2] on val")
        plt.grid(ls="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()

        fig2.savefig(os.path.join(save_dir, "live_z2.png"), dpi=160)

        if snap_every and (epoch % snap_every == 0):
            fig2.savefig(os.path.join(save_dir, f"live_z2_ep{epoch:04d}.png"), dpi=160)

        if show_every and (epoch % show_every == 0):
            plt.show()

        plt.close(fig2)

# ==========================================================
# TRAIN LOOP
# ==========================================================
train_losses, val_losses = [], []
z2_hist = []

SNAP_EVERY = 1
SHOW_EVERY = 0

END_EPOCH = START_EPOCH + EPOCHS - 1
print(f"Training epochs: {START_EPOCH} -> {END_EPOCH}")

for ep in range(START_EPOCH, END_EPOCH + 1):
    tr = train_epoch(model, train_loader)
    va = eval_epoch(model, val_loader)

    if not np.isfinite(va):
        print(f"\nNon-finite val_nll at epoch {ep}. Stopping immediately.")
        print("Keeping previous best checkpoint:", SAVE_PATH)
        print("Recommended: delete bad LAST checkpoint before next resume if it was saved after instability.")
        break

    z2 = per_target_z2(model, val_loader).numpy()

    base_w_pos, phase_w_pos, phase_w_phase = get_phase_weights(model)

    old_lr = get_lr(opt)
    if scheduler is not None:
        scheduler.step(va)
    new_lr = get_lr(opt)

    if new_lr < old_lr:
        print(f"LR reduced: {old_lr:.3e} -> {new_lr:.3e}")

    improved = va < (best_val - MIN_DELTA)

    if improved:
        best_val = va
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    train_losses.append(tr)
    val_losses.append(va)
    z2_hist.append(z2)

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                ep,
                tr,
                va,
                get_lr(opt),
                float(base_w_pos) if base_w_pos is not None else np.nan,
                float(phase_w_pos) if phase_w_pos is not None else np.nan,
                float(phase_w_phase) if phase_w_phase is not None else np.nan,
                epochs_no_improve,
            ]
            + [float(x) for x in z2]
        )

    ckpt = make_checkpoint(
        ep, model, opt, scaler, scheduler, best_val, epochs_no_improve,
        target_names, phase_target_names, phase_target_idx,
        THETA_DIM, P, L, X_mean, X_std, y_mean, y_std,
        base_w_pos, phase_w_pos, phase_w_phase,
    )

    torch.save(ckpt, LAST_PATH)

    if improved:
        torch.save(ckpt, SAVE_PATH)

    _save_live_plots(
        train_losses,
        val_losses,
        z2_hist,
        base_w_pos,
        phase_w_pos,
        phase_w_phase,
        SAVE_DIR,
        ep,
        snap_every=SNAP_EVERY,
        show_every=SHOW_EVERY,
    )

    print(
        f"[{ep:03d}] train_nll={tr:.6f}  val_nll={va:.6f}  "
        f"lr={get_lr(opt):.3e}  no_improve={epochs_no_improve}/{PATIENCE}  "
        f"base_w_pos={base_w_pos}  phase_w_pos={phase_w_pos}  "
        f"phase_w_phase={phase_w_phase}  best={best_val:.6f}"
    )

    if EARLY_STOP and epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {ep}")
        print(f"No validation improvement larger than {MIN_DELTA} for {PATIENCE} epochs.")
        break

print("Done.")
print("Best saved to:", SAVE_PATH)
print("Last saved to:", LAST_PATH)

# ==========================================================
# FINAL LOSS CURVE
# ==========================================================
if len(train_losses) > 0:
    final_loss_path = os.path.join(SAVE_DIR, "loss_curve_final.png")
    base_w_pos, phase_w_pos, phase_w_phase = get_phase_weights(model)

    plt.figure(figsize=(7.4, 4.6))
    plt.plot(train_losses, label="train_nll")
    plt.plot(val_losses, label="val_nll")
    plt.xlabel("epoch index in this run")
    plt.ylabel("NLL")
    plt.grid(ls="--", alpha=0.4)
    plt.legend()

    if base_w_pos is not None and phase_w_pos is not None and phase_w_phase is not None:
        plt.title(
            f"Final | LR={get_lr(opt):.2e} | base_w_pos={base_w_pos:.5f} | "
            f"phase_w_pos={phase_w_pos:.5f} | phase_w_phase={phase_w_phase:.5f}",
            fontsize=8,
        )
    else:
        plt.title(f"Final | LR={get_lr(opt):.2e}", fontsize=10)

    plt.tight_layout()
    plt.savefig(final_loss_path, dpi=160)
    plt.close()

# # ==========================================================
# # SAVE VALIDATION SNAPSHOT — CONSISTENT WITH TRAINING
# # ==========================================================
# VAL_SAVE_PATH = os.path.join(SAVE_DIR, "validation_set.npz")

# print("\nSaving validation snapshot to:")
# print(" ", VAL_SAVE_PATH)

# X_va_clean = X_all_clean[val_r]
# X_va_noisy = X_all[val_r]

# y_va_targets = y_va
# y_va_full = Y_by_psr[0, val_r, :]

# # Always save true phase arrays, even for pred_enc/no_enc.
# phi_tr_save = np.repeat(phase_B[train_r][:, None, :], P, axis=1)
# phi_va_save = np.repeat(phase_B[val_r][:, None, :],   P, axis=1)

# snr_tr = snr_all[train_r]
# snr_va = snr_all[val_r]

# sigma_tr = sigma_all[train_r]
# sigma_va = sigma_all[val_r]

# log10_sigma_tr = log10_sigma_all[train_r]
# log10_sigma_va = log10_sigma_all[val_r]

# np.savez_compressed(
#     VAL_SAVE_PATH,

#     X_va_std=X_va_t.cpu().numpy(),
#     y_va_std=y_va_t.cpu().numpy(),

#     X_va_clean=X_va_clean,
#     X_va_noisy=X_va_noisy,
#     y_va_targets=y_va_targets,
#     y_va_full=y_va_full,

#     phi_va=phi_va_save.astype(np.float32),
#     phi_tr=phi_tr_save.astype(np.float32),

#     X_mean=X_mean.cpu().numpy(),
#     X_std=X_std.cpu().numpy(),
#     y_mean=y_mean.cpu().numpy(),
#     y_std=y_std.cpu().numpy(),

#     snr_va=snr_va,
#     sigma_va=sigma_va,
#     log10_sigma_va=log10_sigma_va,
#     snr_tr=snr_tr,
#     sigma_tr=sigma_tr,
#     log10_sigma_tr=log10_sigma_tr,

#     val_r=val_r,
#     train_r=train_r,
#     param_cols=np.array(param_cols),
#     target_names=np.array(target_names),
#     tidx=np.array(tidx, dtype=np.int64),

#     seed=np.array(SEED),
#     add_noise=np.array(ADD_NOISE),
#     snr_lo=np.array(SNR_LO),
#     snr_hi=np.array(SNR_HI),

#     use_true_phase=np.array(USE_TRUE_PHASE),
#     use_phase_provider=np.array(USE_PHASE_PROVIDER),
#     use_phase=np.array(USE_PHASE),
# )

# print("Saved validation_set.npz successfully.")