#!/usr/bin/env python
# coding: utf-8

"""
hierarchical_cnf.py

HIERARCHICAL MASKED PHASE CNF WITH STANDARD SELF-ATTENTION

Goal
----
- Keep ONE joint posterior / ONE joint continuous normalizing flow (CNF).
- Use the current HierarchicalPTAEncoder from pta_encoder.py.
- Let phase directly affect only selected target parameters.
- Preserve cross-parameter correlations through one shared joint CNF.

Conditioning design
-------------------
    h_base  = HierarchicalPTAEncoder(x, phase=None, use_phasepe=False)
    h_phase = HierarchicalPTAEncoder(x, phase=phase_eff, use_phasepe=True)
    h_delta = h_phase - h_base

CNF vector field
----------------
    f(theta, t) = f_base(theta, h_base, t)
                + phase_gate * phase_mask * f_phase(theta, h_base, h_delta, t)

The explicit phase_gate is important: when phase conditioning is disabled the
phase branch is EXACTLY zero. Merely replacing h_delta by zeros is insufficient,
because a neural network can still output a non-zero value from biases and from
(theta, h_base, t).

For the small posterior dimension used here (D=4 by default), divergence is
computed exactly rather than with a stochastic Hutchinson trace estimator.
This makes validation NLL deterministic and removes trace-estimator noise.

API
---
- THETA_DIM may be overwritten by the run script before PosteriorNet creation.
- PosteriorNet.log_prob(theta, x, phase=None)
- PosteriorNet.sample(n, x, phase=None)
- PosteriorNet.fwd_to_z(theta, x, phase=None)
- PosteriorNet.cond(x, phase=None)

Notes
-----
- Requires torchdiffeq.
- Train CNFs in fp32; AMP/fp16 is intentionally not assumed here.
"""

import math
import torch
import torch.nn as nn
from torchdiffeq import odeint

from pta_encoder import HierarchicalPTAEncoder


# ==========================================================
# GLOBALS
# ==========================================================
THETA_DIM   = 4
CTX_DIM     = 64
FLOW_HIDDEN = 192

# Current hierarchical standard-self-attention defaults.
SA_PATCH   = 20
SA_HEADS   = 8
SA_DMODEL  = 128
SA_DIM_FF  = 256

# CNF solver defaults.
CNF_ATOL      = 1e-3
CNF_RTOL      = 1e-3
CNF_METHOD    = "rk4"
CNF_STEP_SIZE = 0.1


# ==========================================================
# HELPERS
# ==========================================================
def _as_float_mask(indices, D, device=None):
    """Return a (1, D) float mask with ones at selected indices."""
    m = torch.zeros(D, dtype=torch.float32, device=device)
    if indices is not None and len(indices) > 0:
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        if torch.any(idx < 0) or torch.any(idx >= D):
            raise ValueError(
                f"phase_target_idx contains invalid entries for D={D}: {indices}"
            )
        m[idx] = 1.0
    return m.view(1, D)


def _repeat_context(h: torch.Tensor, n: int) -> torch.Tensor:
    """Repeat/truncate a batch of context vectors to exactly n rows."""
    if h.size(0) == n:
        return h
    if h.size(0) == 1:
        return h.repeat(n, 1)
    reps = (n + h.size(0) - 1) // h.size(0)
    return h.repeat(reps, 1)[:n]


# ==========================================================
# CONDITIONER — same current encoder family as hierarchical_dnf.py
# ==========================================================
class SelfAttentionConditioner(nn.Module):
    """
    Hierarchical standard-self-attention conditioner.

    x:     (B, P, L)
    phase: None, (B, L), or (B, P, L)
    h:     (B, ctx_dim)
    """

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


# ==========================================================
# CNF COMPONENTS
# ==========================================================
class CNF_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=FLOW_HIDDEN, depth=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MaskedODEFunc(nn.Module):
    """
    CNF vector field with dimension-selective phase correction.

    Augmented ODE state:
      y:          (B, D)
      logp:       (B,)
      h_base:     (B, ctx_dim)
      h_delta:    (B, ctx_dim)
      phase_gate: (B, 1), exactly 0 or 1

    dy/dt = f_base(y, h_base, t)
          + phase_gate * phase_mask * f_phase(y, h_base, h_delta, t)

    h_base, h_delta, and phase_gate are constant along the ODE trajectory.
    """

    def __init__(
        self,
        theta_dim: int,
        ctx_dim: int,
        hidden: int = FLOW_HIDDEN,
        phase_target_idx=None,
    ):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.ctx_dim = int(ctx_dim)

        self.base_net = CNF_MLP(
            self.theta_dim + self.ctx_dim + 1,
            self.theta_dim,
            hidden=hidden,
            depth=2,
        )

        self.phase_net = CNF_MLP(
            self.theta_dim + 2 * self.ctx_dim + 1,
            self.theta_dim,
            hidden=hidden,
            depth=2,
        )

        self.register_buffer(
            "phase_mask",
            _as_float_mask(phase_target_idx, self.theta_dim),
        )

    def _vector_field(self, y, h_base, h_delta, phase_gate, t):
        B = y.size(0)
        t_feat = torch.ones(B, 1, device=y.device, dtype=y.dtype) * t

        inp_base = torch.cat([y, h_base, t_feat], dim=1)
        f = self.base_net(inp_base)

        inp_phase = torch.cat([y, h_base, h_delta, t_feat], dim=1)
        f_phase = self.phase_net(inp_phase)

        mask = self.phase_mask.to(device=y.device, dtype=y.dtype)
        gate = phase_gate.to(device=y.device, dtype=y.dtype)
        return f + gate * mask * f_phase

    def _exact_divergence(self, f, y):
        """Compute div_y f exactly. Practical and stable for small theta_dim."""
        trace = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)
        for j in range(self.theta_dim):
            grad_j = torch.autograd.grad(
                f[:, j].sum(),
                y,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0][:, j]
            trace = trace + grad_j
        return trace

    def forward(self, t, states):
        y, logp, h_base, h_delta, phase_gate = states

        # Evaluation/sampling may call this function under torch.no_grad().
        # CNF divergence still requires gradients with respect to the ODE state.
        with torch.enable_grad():
            if not y.requires_grad:
                y = y.requires_grad_(True)

            f = self._vector_field(y, h_base, h_delta, phase_gate, t)
            trace = self._exact_divergence(f, y)

        dlogp = -trace
        dh_base = torch.zeros_like(h_base)
        dh_delta = torch.zeros_like(h_delta)
        dphase_gate = torch.zeros_like(phase_gate)

        return f, dlogp, dh_base, dh_delta, dphase_gate



class ConditionalMaskedCNF(nn.Module):
    """Conditional CNF with masked phase-aware vector field."""

    def __init__(
        self,
        theta_dim: int,
        ctx_dim: int = CTX_DIM,
        hidden: int = FLOW_HIDDEN,
        atol: float = CNF_ATOL,
        rtol: float = CNF_RTOL,
        method: str = CNF_METHOD,
        step_size: float = CNF_STEP_SIZE,
        phase_target_idx=None,
    ):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.ctx_dim = int(ctx_dim)

        self.func = MaskedODEFunc(
            theta_dim=self.theta_dim,
            ctx_dim=self.ctx_dim,
            hidden=hidden,
            phase_target_idx=phase_target_idx,
        )
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.method = str(method)
        self.step_size = float(step_size)

    def _base_logprob(self, z):
        return (
            -0.5 * (z ** 2).sum(dim=1)
            -0.5 * self.theta_dim * math.log(2 * math.pi)
        )

    @staticmethod
    def _zero_delta(h_base):
        return torch.zeros_like(h_base)

    @staticmethod
    def _phase_gate(h_base, enabled: bool):
        value = 1.0 if enabled else 0.0
        return torch.full(
            (h_base.size(0), 1),
            value,
            device=h_base.device,
            dtype=h_base.dtype,
        )

    def _ode_kwargs(self):
        fixed_step_methods = {
            "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"
        }
        kwargs = {
            "atol": self.atol,
            "rtol": self.rtol,
            "method": self.method,
        }
        if self.method in fixed_step_methods:
            kwargs["options"] = {"step_size": self.step_size}
        return kwargs

    def _odeint(self, y0, t_span):
        return odeint(self.func, y0, t_span, **self._ode_kwargs())

    def _state_only_rhs(self, t, states):
        y, h_base, h_delta, phase_gate = states
        f = self.func._vector_field(y, h_base, h_delta, phase_gate, t)
        return (
            f,
            torch.zeros_like(h_base),
            torch.zeros_like(h_delta),
            torch.zeros_like(phase_gate),
        )

    def _odeint_state_only(self, y0, t_span):
        return odeint(self._state_only_rhs, y0, t_span, **self._ode_kwargs())

    def _prepare_augmented_state(self, y, h_base, h_delta):
        if y.ndim != 2:
            raise ValueError(f"theta/z must have shape (B, D), got {tuple(y.shape)}")
        if h_base.ndim != 2:
            raise ValueError(
                f"h_base must have shape (B, ctx_dim), got {tuple(h_base.shape)}"
            )
        if y.size(0) != h_base.size(0):
            raise ValueError(
                f"Batch mismatch: state batch={y.size(0)} but context batch={h_base.size(0)}"
            )
        if y.size(1) != self.theta_dim:
            raise ValueError(
                f"Expected state dimension D={self.theta_dim}, got {y.size(1)}"
            )

        phase_enabled = h_delta is not None
        if h_delta is None:
            h_delta_eff = self._zero_delta(h_base)
        else:
            if h_delta.shape != h_base.shape:
                raise ValueError(
                    f"h_delta shape {tuple(h_delta.shape)} must match "
                    f"h_base {tuple(h_base.shape)}"
                )
            h_delta_eff = h_delta

        gate = self._phase_gate(h_base, enabled=phase_enabled)
        return h_delta_eff, gate

    def fwd_to_z(self, theta, h_base, h_delta=None):
        """
        Map theta(t=1) -> z(t=0), returning (z, delta_logp).

        delta_logp is the reverse-time accumulated log-density correction used by
        log_prob below.
        """
        B = theta.size(0)
        h_delta_eff, gate = self._prepare_augmented_state(theta, h_base, h_delta)

        t_span = torch.tensor(
            [1.0, 0.0],
            device=theta.device,
            dtype=theta.dtype,
        )
        logp0 = torch.zeros(B, device=theta.device, dtype=theta.dtype)

        yt, logpt, _, _, _ = self._odeint(
            (theta, logp0, h_base, h_delta_eff, gate),
            t_span,
        )

        return yt[-1], logpt[-1]

    @torch.no_grad()
    def to_base(self, theta, h_base, h_delta=None):
        """Map theta(t=1) -> z(t=0) without computing the CNF divergence."""
        h_delta_eff, gate = self._prepare_augmented_state(theta, h_base, h_delta)
        t_span = torch.tensor(
            [1.0, 0.0],
            device=theta.device,
            dtype=theta.dtype,
        )
        zT, _, _, _ = self._odeint_state_only(
            (theta, h_base, h_delta_eff, gate),
            t_span,
        )
        return zT[-1]

    def log_prob(self, theta, h_base, h_delta=None):
        """Return log p(theta | context)."""
        z, delta_logp = self.fwd_to_z(theta, h_base, h_delta=h_delta)
        return self._base_logprob(z) - delta_logp

    @torch.no_grad()
    def sample(self, n, h_base, h_delta=None):
        """Sample theta ~ p(theta | x) by integrating z(t=0) -> theta(t=1)."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        H_base = _repeat_context(h_base, n)
        H_delta = None if h_delta is None else _repeat_context(h_delta, n)

        z0 = torch.randn(
            n,
            self.theta_dim,
            device=H_base.device,
            dtype=H_base.dtype,
        )
        H_delta_eff, gate = self._prepare_augmented_state(z0, H_base, H_delta)

        t_span = torch.tensor(
            [0.0, 1.0],
            device=H_base.device,
            dtype=H_base.dtype,
        )

        # Sampling does not need the divergence/log-density, so avoid paying for
        # exact trace computation at every ODE evaluation.
        yT, _, _, _ = self._odeint_state_only(
            (z0, H_base, H_delta_eff, gate),
            t_span,
        )
        return yT[-1]


# ==========================================================
# POSTERIOR NETWORK
# ==========================================================
class PosteriorNet(nn.Module):
    """
    Full hierarchical CNF posterior with standard self-attention conditioning.

    h_base  = cond_base(x, None)
    h_phase = cond_phase(x, phase)
    h_delta = h_phase - h_base

    The CNF always receives h_base. When use_phase=True, h_delta supplies a
    dimension-masked phase correction. When use_phase=False, the explicit phase
    gate is zero, so the phase branch contributes exactly nothing.
    """

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
        cnf_hidden: int = FLOW_HIDDEN,
        atol: float = CNF_ATOL,
        rtol: float = CNF_RTOL,
        method: str = CNF_METHOD,
        step_size: float = CNF_STEP_SIZE,
        target_names=None,
        phase_target_names=("log10_n", "q", "e0", "log10_M"),
    ):
        super().__init__()

        self.use_phase = bool(use_phase)
        self.phase_provider = phase_provider
        self.target_names = list(target_names) if target_names is not None else None
        self.phase_target_names = (
            list(phase_target_names) if phase_target_names is not None else []
        )

        if self.target_names is None:
            raise ValueError(
                "PosteriorNet requires target_names to build phase_target_idx "
                "for phase-conditioned CNF modeling."
            )

        if len(self.target_names) != THETA_DIM:
            raise ValueError(
                f"len(target_names)={len(self.target_names)} but THETA_DIM={THETA_DIM}. "
                "Set hierarchical_cnf.THETA_DIM before constructing PosteriorNet."
            )

        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError(f"target_names must be unique, got {self.target_names}")

        name_to_idx = {name: i for i, name in enumerate(self.target_names)}
        missing = [
            name for name in self.phase_target_names
            if name not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"phase_target_names not found in target_names: {missing}. "
                f"target_names={self.target_names}"
            )

        self.phase_target_idx = [
            name_to_idx[name] for name in self.phase_target_names
        ]

        # Keep the same two-conditioner design as the current hierarchical DNF.
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

        self.flow = ConditionalMaskedCNF(
            theta_dim=THETA_DIM,
            ctx_dim=ctx_dim,
            hidden=cnf_hidden,
            atol=atol,
            rtol=rtol,
            method=method,
            step_size=step_size,
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
        """Compatibility helper returning the effective context representation."""
        h_base, h_delta = self._contexts(x, phase)
        return h_base if h_delta is None else h_base + h_delta

    def fwd_to_z(self, theta, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.fwd_to_z(theta, h_base, h_delta=h_delta)

    @torch.no_grad()
    def to_base(self, theta, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.to_base(theta, h_base, h_delta=h_delta)

    def log_prob(self, theta, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.log_prob(theta, h_base, h_delta=h_delta)

    @torch.no_grad()
    def sample(self, n, x, phase=None):
        h_base, h_delta = self._contexts(x, phase)
        return self.flow.sample(n, h_base, h_delta=h_delta)
