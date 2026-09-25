#!/usr/bin/env python
# coding: utf-8

"""
hierarchical_dnf.py

HIERARCHICAL MASKED PHASE DNF WITH STANDARD SELF-ATTENTION

Goal:
- Keep ONE joint posterior / ONE joint RealNVP flow.
- Use HierarchicalPTAEncoder with standard multi-head self-attention.
- Let phase directly affect ONLY selected target parameters.
- Preserve cross-parameter correlations through the shared joint flow.

Design:
- h_base  = HierarchicalPTAEncoder(x, phase=None, use_phasepe=False)
- h_phase = HierarchicalPTAEncoder(x, phase=phase_eff, use_phasepe=True)
- h_delta = h_phase - h_base

Each affine coupling layer always receives h_base.
Only transformed output dimensions whose indices are in phase_target_idx
receive the additional h_delta contribution.
"""

import math
import torch
import torch.nn as nn

THETA_DIM   = 4
CTX_DIM     = 64
FLOW_LAYERS = 10
FLOW_HIDDEN = 128

SA_PATCH   = 20
SA_HEADS   = 8
SA_DMODEL  = 128
SA_DEPTH   = 4
SA_DIM_FF  = 256

from pta_encoder import HierarchicalPTAEncoder


def _as_bool_mask(indices, D, device=None):
    m = torch.zeros(D, dtype=torch.bool, device=device)
    if indices is not None and len(indices) > 0:
        m[torch.as_tensor(indices, dtype=torch.long, device=device)] = True
    return m


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=FLOW_HIDDEN, depth=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, D, keep_idx, ctx_dim=CTX_DIM, hidden=FLOW_HIDDEN, phase_target_idx=None):
        super().__init__()
        self.D = D
        self.keep_idx = list(keep_idx)
        self.trans_idx = [i for i in range(D) if i not in self.keep_idx]

        phase_target_set = set(phase_target_idx or [])
        self.register_buffer("phase_target_mask_full", _as_bool_mask(phase_target_idx, D))
        local_mask = [1.0 if idx in phase_target_set else 0.0 for idx in self.trans_idx]
        self.register_buffer("phase_target_mask_local", torch.tensor(local_mask, dtype=torch.float32).view(1, -1))

        in_dim_base = len(self.keep_idx) + ctx_dim
        self.s_net_base = MLP(in_dim_base, len(self.trans_idx), hidden=hidden, depth=2)
        self.t_net_base = MLP(in_dim_base, len(self.trans_idx), hidden=hidden, depth=2)

        in_dim_phase = len(self.keep_idx) + ctx_dim + ctx_dim
        self.s_net_phase = MLP(in_dim_phase, len(self.trans_idx), hidden=hidden, depth=2)
        self.t_net_phase = MLP(in_dim_phase, len(self.trans_idx), hidden=hidden, depth=2)

    def forward(self, theta, h_base, h_delta=None, reverse=False):
        x_base = torch.cat([theta[:, self.keep_idx], h_base], dim=1)
        s = self.s_net_base(x_base)
        t = self.t_net_base(x_base)

        if h_delta is not None:
            x_phase = torch.cat([theta[:, self.keep_idx], h_base, h_delta], dim=1)
            s = s + self.s_net_phase(x_phase) * self.phase_target_mask_local
            t = t + self.t_net_phase(x_phase) * self.phase_target_mask_local

        out = theta.clone()
        if not reverse:
            out[:, self.trans_idx] = theta[:, self.trans_idx] * torch.exp(s) + t
            logdet = s.sum(dim=1)
        else:
            out[:, self.trans_idx] = (theta[:, self.trans_idx] - t) * torch.exp(-s)
            logdet = (-s).sum(dim=1)
        return out, logdet


def default_masks_D(D, n_layers):
    pairs = []
    base = list(range(D))
    for i in range(D * 3):
        a = base[i % D]
        b = base[(i + 1) % D]
        if a != b:
            pairs.append(sorted(list({a, b})))
    uniq, seen = [], set()
    for p in pairs:
        t = tuple(p)
        if t not in seen:
            uniq.append(p)
            seen.add(t)
        if len(uniq) >= n_layers:
            break
    return uniq[:n_layers]


class ConditionalRealNVP(nn.Module):
    def __init__(self, D=THETA_DIM, ctx_dim=CTX_DIM, n_layers=FLOW_LAYERS, hidden=FLOW_HIDDEN, phase_target_idx=None):
        super().__init__()
        masks = default_masks_D(D, n_layers)
        self.D = D
        self.layers = nn.ModuleList([
            AffineCoupling(D, keep_idx=m, ctx_dim=ctx_dim, hidden=hidden, phase_target_idx=phase_target_idx)
            for m in masks
        ])

    def fwd_to_z(self, theta, h_base, h_delta=None):
        logdet = torch.zeros(theta.size(0), device=theta.device)
        x = theta
        for g in self.layers:
            x, ld = g(x, h_base, h_delta=h_delta, reverse=False)
            logdet = logdet + ld
        return x, logdet

    def inv_from_z(self, z, h_base, h_delta=None):
        logdet = torch.zeros(z.size(0), device=z.device)
        x = z
        for g in reversed(self.layers):
            x, ld = g(x, h_base, h_delta=h_delta, reverse=True)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, theta, h_base, h_delta=None):
        z, logdet = self.fwd_to_z(theta, h_base, h_delta=h_delta)
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.D * math.log(2 * math.pi)
        return log_pz + logdet

    @torch.no_grad()
    def sample(self, n, h_base, h_delta=None):
        if h_base.size(0) == 1:
            H_base = h_base.repeat(n, 1)
            H_delta = None if h_delta is None else h_delta.repeat(n, 1)
        elif h_base.size(0) != n:
            reps = (n + h_base.size(0) - 1) // h_base.size(0)
            H_base = h_base.repeat(reps, 1)[:n]
            H_delta = None if h_delta is None else h_delta.repeat(reps, 1)[:n]
        else:
            H_base = h_base
            H_delta = h_delta
        z = torch.randn(n, self.D, device=H_base.device)
        theta, _ = self.inv_from_z(z, H_base, h_delta=H_delta)
        return theta


class SelfAttentionConditioner(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_pulsars: int,
        ctx_dim: int = CTX_DIM,
        *,
        patch: int = SA_PATCH,
        d_model: int = SA_DMODEL,
        temporal_depth: int = 2,
        cross_depth: int = 2,
        dim_ff: int = SA_DIM_FF,
        heads: int = SA_HEADS,
        p_drop: float = 0.1,
        use_phase: bool = False,
        phase_provider=None,
        x_mean=None,
        x_std=None,
        weighting: str = "learned",
        alpha_pos: float = 1.0,
        alpha_phase: float = 1.0,
        phase_pool: str = "mean",
    ):
        super().__init__()
        self.net = HierarchicalPTAEncoder(
            seq_len=seq_len,
            n_pulsars=n_pulsars,
            out_dim=ctx_dim,
            patch=patch,
            d_model=d_model,
            temporal_depth=temporal_depth,
            cross_depth=cross_depth,
            dim_ff=dim_ff,
            heads=heads,
            p_drop=p_drop,
            use_posenc=True,
            use_phasepe=use_phase,
            phase_provider=phase_provider,
            x_mean=x_mean,
            x_std=x_std,
            weighting=weighting,
            alpha_pos=alpha_pos,
            alpha_phase=alpha_phase,
            phase_pool=phase_pool,
        )

    def forward(self, x, phase=None):
        return self.net(x, phase=phase)


class PosteriorNet(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_pulsars: int,
        *,
        use_phase: bool = False,
        phase_provider=None,
        x_mean=None,
        x_std=None,
        ctx_dim: int = CTX_DIM,
        target_names=None,
        phase_target_names=("log10_n", "q", "e0", "log10_M"),
    ):
        super().__init__()
        self.use_phase = bool(use_phase)
        self.phase_provider = phase_provider
        self.target_names = list(target_names) if target_names is not None else None
        self.phase_target_names = list(phase_target_names) if phase_target_names is not None else []

        if self.target_names is None:
            raise ValueError("PosteriorNet requires target_names to build phase_target_idx for masked phase modeling.")

        name_to_idx = {n: i for i, n in enumerate(self.target_names)}
        missing = [n for n in self.phase_target_names if n not in name_to_idx]
        if missing:
            raise ValueError(f"phase_target_names not found in target_names: {missing}. target_names={self.target_names}")
        self.phase_target_idx = [name_to_idx[n] for n in self.phase_target_names]

        self.cond_base = SelfAttentionConditioner(
            seq_len=seq_len,
            n_pulsars=n_pulsars,
            ctx_dim=ctx_dim,
            use_phase=False,
            phase_provider=None,
            x_mean=x_mean,
            x_std=x_std,
            alpha_phase=0.0,
        )

        self.cond_phase = SelfAttentionConditioner(
            seq_len=seq_len,
            n_pulsars=n_pulsars,
            ctx_dim=ctx_dim,
            use_phase=self.use_phase,
            phase_provider=phase_provider,
            x_mean=x_mean,
            x_std=x_std,
            alpha_phase=1.0,
        )

        self.flow = ConditionalRealNVP(
            D=THETA_DIM,
            ctx_dim=ctx_dim,
            n_layers=FLOW_LAYERS,
            hidden=FLOW_HIDDEN,
            phase_target_idx=self.phase_target_idx,
        )

    def _contexts(self, x, phase=None):
        h_base = self.cond_base(x, None)
        if not self.use_phase:
            return h_base, None
        h_phase = self.cond_phase(x, phase)
        h_delta = h_phase - h_base
        return h_base, h_delta

    def cond(self, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return h_base if h_delta is None else h_base + h_delta

    def fwd_to_z(self, theta, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.fwd_to_z(theta, h_base, h_delta=h_delta)

    def log_prob(self, theta, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.log_prob(theta, h_base, h_delta=h_delta)

    @torch.no_grad()
    def sample(self, n, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.sample(n, h_base, h_delta=h_delta)
