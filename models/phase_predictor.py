#!/usr/bin/env python
# coding: utf-8
"""
phase_pred.py

Inference utilities for realisation-mode phase + SNR model.

Input:
    x : (P,L) or (B,P,L), unstandardized PTA realisation

Outputs:
    phi_pred : (B,L)
    snr_pred : (B,)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn


# ==========================================================
# Positional Encoding
# ==========================================================
class SinusoidalPE(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


def _normalize_unit(y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return y / torch.clamp(torch.linalg.norm(y, dim=-1, keepdim=True), min=eps)


# ==========================================================
# Slow full-resolution model
# ==========================================================
class PhaseTransformerRealisation(nn.Module):
    def __init__(
        self,
        P: int,
        L: int,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        d_ff: int = 512,
        p_drop: float = 0.1,
    ):
        super().__init__()

        self.P = P
        self.L = L

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

    def forward(self, x: torch.Tensor):
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


# ==========================================================
# Fast downsampled model
# ==========================================================
class PhaseTransformerRealisationFast(nn.Module):
    def __init__(
        self,
        P: int,
        L: int,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        d_ff: int = 512,
        p_drop: float = 0.1,
        ds_stride: int = 4,
    ):
        super().__init__()

        assert L % ds_stride == 0

        self.P = P
        self.L = L
        self.S = ds_stride
        self.Ls = L // ds_stride

        self.in_proj = nn.Linear(P, d_model)

        self.ds_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=ds_stride, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
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

    def forward(self, x: torch.Tensor):
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


# ==========================================================
# Bundle
# ==========================================================
@dataclass
class PhaseModelBundle:
    model: nn.Module
    mu_x: torch.Tensor
    std_x: torch.Tensor
    P: int
    L: int
    use_fast: bool
    ds_stride: int

    def to(self, device: torch.device):
        self.model.to(device)
        self.mu_x = self.mu_x.to(device)
        self.std_x = self.std_x.to(device)
        return self


def _build_model_from_ckpt(ckpt):
    P = int(ckpt["P"])
    L = int(ckpt["L"])

    use_fast = bool(ckpt.get("USE_FAST", True))
    ds_stride = int(ckpt.get("DS_STRIDE", 4))

    d_model = int(ckpt.get("D_MODEL", 128))
    depth = int(ckpt.get("DEPTH", 4))
    heads = int(ckpt.get("HEADS", 4))
    d_ff = int(ckpt.get("D_FF", 512))
    p_drop = float(ckpt.get("P_DROP", 0.1))

    if use_fast:
        model = PhaseTransformerRealisationFast(
            P=P,
            L=L,
            d_model=d_model,
            depth=depth,
            heads=heads,
            d_ff=d_ff,
            p_drop=p_drop,
            ds_stride=ds_stride,
        )
    else:
        model = PhaseTransformerRealisation(
            P=P,
            L=L,
            d_model=d_model,
            depth=depth,
            heads=heads,
            d_ff=d_ff,
            p_drop=p_drop,
        )

    return model, P, L, use_fast, ds_stride


# ==========================================================
# Load model
# ==========================================================
def load_phase_model(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> PhaseModelBundle:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    if map_location is None:
        map_location = device

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    model, P, L, use_fast, ds_stride = _build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    mu_x = ckpt["mu_x"].detach().float()
    std_x = ckpt["std_x"].detach().float()

    return PhaseModelBundle(
        model=model,
        mu_x=mu_x,
        std_x=std_x,
        P=P,
        L=L,
        use_fast=use_fast,
        ds_stride=ds_stride,
    ).to(device)


# ==========================================================
# Predict phase + SNR
# ==========================================================
@torch.no_grad()
def predict_phase_and_snr(
    x: torch.Tensor,
    bundle: PhaseModelBundle,
    assume_standardized: bool = False,
    return_unit: bool = False,
):
    """
    Args:
        x:
            (P,L) or (B,P,L)
        assume_standardized:
            False if x is raw/unstandardized.
            True if x is already standardized with ckpt mu_x/std_x.
        return_unit:
            Also return unit-vector phase representation.

    Returns:
        phi_pred:
            (B,L), wrapped phase in radians
        snr_pred:
            (B,), linear realisation-level SNR
        optional y_unit:
            (B,L,2)
    """

    if x.dim() == 2:
        x = x.unsqueeze(0)

    if x.dim() != 3:
        raise ValueError(f"x must be (P,L) or (B,P,L), got {tuple(x.shape)}")

    B, P, L = x.shape

    if P != bundle.P or L != bundle.L:
        raise ValueError(
            f"Input shape (P,L)=({P},{L}) does not match checkpoint ({bundle.P},{bundle.L})"
        )

    x = x.to(bundle.mu_x.device, dtype=torch.float32)

    if not assume_standardized:
        x_flat = x.reshape(B, P * L)
        x_flat = (x_flat - bundle.mu_x) / bundle.std_x
        x = x_flat.reshape(B, P, L)

    y_phase, snr_log10_pred = bundle.model(x)

    y_unit = _normalize_unit(y_phase)
    phi_pred = torch.atan2(y_unit[..., 1], y_unit[..., 0])

    snr_pred = torch.pow(10.0, snr_log10_pred)

    if return_unit:
        return phi_pred, snr_pred, y_unit

    return phi_pred, snr_pred


@torch.no_grad()
def predict_phase(
    x: torch.Tensor,
    bundle: PhaseModelBundle,
    assume_standardized: bool = False,
    return_unit: bool = False,
):
    """
    Backward-compatible phase-only wrapper.
    """

    out = predict_phase_and_snr(
        x,
        bundle,
        assume_standardized=assume_standardized,
        return_unit=return_unit,
    )

    if return_unit:
        phi, snr, y_unit = out
        return phi, y_unit

    phi, snr = out
    return phi


# ==========================================================
# PhaseProvider for SBI model
# ==========================================================
class PhaseProvider(nn.Module):
    """
    Phase provider for EA/SBI conditioning.

    Input:
        standardized x from SBI model:
            (B,P,L), (B,P*L), or (B,L)

    Output:
        phase with same PTA layout:
            (B,P,L) for (B,P,L)
            (B,P*L) for (B,P*L)
            (B,L) for (B,L)

    Also stores:
        self.last_snr_pred
            predicted realisation-level SNR, shape (B,)
    """

    def __init__(
        self,
        phase_ckpt_path: str,
        *,
        device: str | torch.device = "cpu",
        base_len: int | None = None,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.bundle = load_phase_model(phase_ckpt_path, device=self.device)

        self.base_len = int(base_len) if base_len is not None else int(self.bundle.L)

        self.x_mean = None
        self.x_std = None

        self.last_snr_pred = None
        self.last_snr_log10_pred = None

    def _as_2d_stats(self, t: torch.Tensor, D: int) -> torch.Tensor:
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)

        if t.ndim == 1:
            t = t.view(1, -1)

        if t.shape[1] != D:
            raise ValueError(
                f"Stats length mismatch: got {tuple(t.shape)}, need (1,{D})"
            )

        return t

    def _unstandardize_ea(self, x_std_in: torch.Tensor, D_flat: int) -> torch.Tensor:
        if self.x_mean is None or self.x_std is None:
            raise ValueError(
                "PhaseProvider: x_mean/x_std not set. "
                "Set phase_provider.x_mean = X_mean and phase_provider.x_std = X_std."
            )

        mu = self._as_2d_stats(self.x_mean, D_flat)
        sd = self._as_2d_stats(self.x_std, D_flat)

        return x_std_in * sd + mu

    def _infer_P_from_flat(self, D_flat: int) -> int:
        if D_flat % self.base_len != 0:
            raise ValueError(
                f"Cannot infer P: D_flat={D_flat} not divisible by L={self.base_len}"
            )
        return D_flat // self.base_len

    @torch.no_grad()
    def forward(self, x_std_in: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x_std_in, device=self.device, dtype=torch.float32)

        # --------------------------------------------------
        # Case 1: flattened input (B, P*L) or single-pulsar (B,L)
        # --------------------------------------------------
        if x.ndim == 2:
            B, D = x.shape

            # Flattened full PTA: (B,P*L)
            if D != self.base_len:
                P = self._infer_P_from_flat(D)

                x_unstd_flat = self._unstandardize_ea(x, D)
                x_unstd = x_unstd_flat.view(B, P, self.base_len)

                phi_BL, snr_B = predict_phase_and_snr(
                    x_unstd,
                    self.bundle,
                    assume_standardized=False,
                )

                self.last_snr_pred = snr_B.detach()
                self.last_snr_log10_pred = torch.log10(torch.clamp(snr_B, min=1e-8))

                phi_rep = phi_BL.unsqueeze(1).expand(B, P, self.base_len)
                return phi_rep.reshape(B, P * self.base_len)

            # Single-pulsar style input: (B,L)
            # Replicate across P because phase model expects full PTA realisation.
            P = int(self.bundle.P)

            # For this case, stats are only length L if EA passes per-pulsar input.
            # If your SBI always passes (B,P,L) or (B,P*L), this branch is rarely used.
            x_unstd_L = self._unstandardize_ea(x, self.base_len)
            x_unstd = x_unstd_L.unsqueeze(1).expand(B, P, self.base_len)

            phi_BL, snr_B = predict_phase_and_snr(
                x_unstd,
                self.bundle,
                assume_standardized=False,
            )

            self.last_snr_pred = snr_B.detach()
            self.last_snr_log10_pred = torch.log10(torch.clamp(snr_B, min=1e-8))

            return phi_BL

        # --------------------------------------------------
        # Case 2: realisation input (B,P,L)
        # --------------------------------------------------
        if x.ndim == 3:
            B, P, L = x.shape

            if L != self.base_len:
                raise ValueError(f"Expected L={self.base_len}, got L={L}")

            x_flat = x.reshape(B, P * L)
            x_unstd_flat = self._unstandardize_ea(x_flat, P * L)
            x_unstd = x_unstd_flat.view(B, P, L)

            phi_BL, snr_B = predict_phase_and_snr(
                x_unstd,
                self.bundle,
                assume_standardized=False,
            )

            self.last_snr_pred = snr_B.detach()
            self.last_snr_log10_pred = torch.log10(torch.clamp(snr_B, min=1e-8))

            phi_BPL = phi_BL.unsqueeze(1).expand(B, P, L)

            return phi_BPL

        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")