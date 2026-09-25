# pta_encoder.py
# - Uses standard multi-head self-attention throughout.
# - SelfAttentionUnifiedPE accepts x shaped (B,L) or (B,P,L) and flattens to (B,P*L).
# - PhaseProvider accepts x shaped (B,L), (B,P,L), or (B,P*L).
# - If PhaseProvider returns SNR per pulsar (B,P), it is averaged to (B,).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from phase_predictor import predict_phase, load_phase_model, PhaseProvider

# --------------------------- Sinusoidal Positional Encodings -----------------
class SinusoidalPE(nn.Module):
    """Classic index-based sinusoidal PE table; you add it to tokens."""
    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (S,1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (S,d)


class PhaseSinusoidalPE(nn.Module):
    """
    Phase-based sinusoidal positional encoding.

    IMPORTANT:
    - If input phase is unwrapped, wrap it into [-π, π] so sin/cos stay stable.

    Input:
        phase_tok: (B, S)
    Output:
        (B, S, d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        div = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        self.register_buffer("div_term", torch.exp(div))
        self.d_model = d_model

    def forward(self, phase_tok: torch.Tensor) -> torch.Tensor:
        B, S = phase_tok.shape
        phase_wrapped = (phase_tok + math.pi) % (2 * math.pi) - math.pi
        angles = phase_wrapped.unsqueeze(-1) * self.div_term.view(1, 1, -1)
        pe = torch.zeros(B, S, self.d_model, device=phase_tok.device, dtype=phase_tok.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe


class PerPulsarTemporalEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        patch: int = 20,
        depth: int = 2,
        dim_ff: int = 256,
        heads: int = 8,
        p_drop: float = 0.1,
        use_posenc: bool = True,
        use_phasepe: bool = False,
        phase_pool: str = "mean",
        weighting: str = "learned",
        alpha_pos: float = 1.0,
        alpha_phase: float = 1.0,
    ):
        super().__init__()
        assert weighting in {"manual", "learned", "hybrid"}

        self.patch = patch
        self.use_posenc = use_posenc
        self.use_phasepe = use_phasepe
        self.phase_pool = phase_pool
        self.weighting = weighting
        self.alpha_pos = float(alpha_pos)
        self.alpha_phase = float(alpha_phase)

        self.embed = PatchEmbed1D(patch, d_model, dropout=p_drop)

        if self.use_posenc:
            self.pos_pe = SinusoidalPE(d_model, max_len=1000)

        if self.use_phasepe:
            self.phase_pe = PhaseSinusoidalPE(d_model)

        if self.use_posenc:
            self.w_pos = (
                nn.Parameter(torch.tensor(self.alpha_pos))
                if weighting in {"learned", "hybrid"}
                else torch.tensor(self.alpha_pos)
            )
        else:
            self.w_pos = None

        if self.use_phasepe:
            self.w_phase = (
                nn.Parameter(torch.tensor(self.alpha_phase))
                if weighting in {"learned", "hybrid"}
                else torch.tensor(self.alpha_phase)
            )
        else:
            self.w_phase = None

        self.blocks = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model, dim_ff, heads=heads, p_drop=p_drop
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _pool_phase_by_patch(self, phase: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        phase: (B*P, L)
        returns: (B*P, S)
        """
        if phase.ndim == 3:
            phase = phase.squeeze(-1)

        if phase.size(1) % patch_size != 0:
            phase = F.pad(phase, (0, patch_size - (phase.size(1) % patch_size)))

        chunks = phase.view(phase.size(0), -1, patch_size)   # (B*P, S, patch)
        if self.phase_pool == "mean":
            return chunks.mean(dim=-1)
        else:
            return chunks[:, :, patch_size // 2]

    def forward(self, x_bpL: torch.Tensor, phase_bpL: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_bpL:     (B*P, L)
        phase_bpL: (B*P, L) or None
        """
        z, (_B, _L, Ppatch) = self.embed(x_bpL)   # (B*P, S_t, d)
        S_t = z.size(1)

        if self.use_posenc:
            pos_term = self.pos_pe.pe[:S_t].unsqueeze(0).to(z.dtype)
            z = z + (self.w_pos * pos_term if isinstance(self.w_pos, nn.Parameter)
                     else float(self.w_pos) * pos_term)

        if self.use_phasepe:
            if phase_bpL is None:
                raise ValueError("use_phasepe=True but phase_bpL=None in PerPulsarTemporalEncoder")

            phase_tok = self._pool_phase_by_patch(phase_bpL, Ppatch)   # (B*P, S_t)
            phi_term = self.phase_pe(phase_tok)                        # (B*P, S_t, d)

            z = z + (self.w_phase * phi_term if isinstance(self.w_phase, nn.Parameter)
                     else float(self.w_phase) * phi_term)

        for blk in self.blocks:
            z = blk(z)

        z = self.norm(z).mean(dim=1)   # (B*P, d)
        return z



        


class CrossPulsarEncoder(nn.Module):
    def __init__(
        self,
        n_pulsars: int = 10,
        d_model: int = 128,
        depth: int = 2,
        dim_ff: int = 256,
        heads: int = 8,
        p_drop: float = 0.1,
        use_pulsar_posenc: bool = True,
    ):
        super().__init__()
        self.use_pulsar_posenc = use_pulsar_posenc

        if use_pulsar_posenc:
            self.pulsar_pe = SinusoidalPE(d_model, max_len=n_pulsars)

        self.blocks = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model, dim_ff, heads=heads, p_drop=p_drop
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, P, d)
        P = z.size(1)
        if self.use_pulsar_posenc:
            z = z + self.pulsar_pe.pe[:P].unsqueeze(0).to(z.dtype)

        for blk in self.blocks:
            z = blk(z)

        z = self.norm(z).mean(dim=1)   # (B, d)
        return z





            
# ------------------------------ Patch Embedding ------------------------------
class PatchEmbed1D(nn.Module):
    """(B,L) -> tokens (B,S,d_model) with Linear(patch->d_model)."""
    def __init__(self, patch: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(-1)  # (B,L,1)->(B,L)
        B, L = x.shape
        P = self.patch
        if L % P != 0:
            x = F.pad(x, (0, P - (L % P)))
            L = x.shape[1]
        tok = x.view(B, L // P, P)   # (B,S,P)
        tok = self.proj(tok)         # (B,S,d)
        return self.drop(F.gelu(tok)), (B, L, P)


# ------------------------- Multi-Head Self-Attention --------------------------
class SelfAttentionEncoderLayer(nn.Module):
    """Standard Transformer encoder layer using multi-head self-attention.

    Input/output shape: ``(B, S, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        heads: int = 4,
        p_drop: float = 0.1,
    ):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=p_drop,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(dim_ff, d_model),
        )
        self.drop_ff = nn.Dropout(p_drop)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop_attn(attn_out))
        x = self.norm2(x + self.drop_ff(self.ff(x)))
        return x


# ------------------------------ CNN Stem -------------------------------------
class CNNStem1D(nn.Module):
    """
    Two-layer 1D CNN stem producing tokens (B,S,d_model) from (B,L).
    """
    def __init__(self, d_model: int, *, k1: int = 7, k2: int = 5,
                 pool: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=k1, stride=2, padding=k1//2)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(pool)
        self.drop1 = nn.Dropout(p_drop)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=k2, stride=2, padding=k2//2)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(pool)
        self.drop2 = nn.Dropout(p_drop)

        self.proj  = nn.Conv1d(128, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        x = x.unsqueeze(1)  # (B,1,L)
        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.proj(x)      # (B,d_model,S)
        x = x.transpose(1, 2) # (B,S,d_model)
        return x


# -------------------- Self-Attention with Unified PEs ------------------------
class SelfAttentionUnifiedPE(nn.Module):
    """
    Self-attention encoder with:
      - optional sinusoidal index PE
      - optional phase PE (either provided or via PhaseProvider)
      - optional SNR encoding (provided or predicted by PhaseProvider)

    UPDATED: x can be (B,L) OR (B,P,L). If (B,P,L), it flattens to (B,P*L).
    Phase can be (B,L) or (B,P,L); it is flattened to match x if needed.
    """
    def __init__(self, seq_len: int, out_dim: int, *,
                 patch: int = 16, d_model: int = 128, depth: int = 4, dim_ff: int = 256,
                 heads: int = 4, p_drop: float = 0.1,
                 use_posenc: bool = True, use_phasepe: bool = True,
                 weighting: str = "learned",
                 alpha_pos: float = 1.0, alpha_phase: float = 1.0,
                 phase_pool: str = "mean",
                 cnn_stem: bool = True, stem_k1: int = 7, stem_k2: int = 5, stem_pool: int = 2,
                 phase_provider: PhaseProvider | None = None,
                 x_mean: torch.Tensor | None = None,
                 x_std:  torch.Tensor | None = None,
                 use_snrenc: bool = False,
                 alpha_snr: float = 1.0):
        super().__init__()
        assert weighting in {"manual", "learned", "hybrid"}
        self.patch = patch
        self.phase_pool = phase_pool
        self.use_pos, self.use_phase = use_posenc, use_phasepe
        self.weighting = weighting
        self.alpha_pos, self.alpha_phase = float(alpha_pos), float(alpha_phase)
        self.cnn_stem = cnn_stem
        self.use_snr = bool(use_snrenc)

        self.phase_provider = phase_provider
        self.register_buffer("x_mean_buf", torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach())
        self.register_buffer("x_std_buf",  torch.as_tensor(1.0) if x_std  is None else x_std.clone().detach())

        # token embedding
        if cnn_stem:
            self.stem = CNNStem1D(d_model, k1=stem_k1, k2=stem_k2, pool=stem_pool, p_drop=p_drop)
            self.embed = None
        else:
            self.embed = PatchEmbed1D(patch, d_model, dropout=p_drop)
            self.stem  = None

        # PEs
        MAX_TOKENS = 20000
        
        if self.use_pos:
            self.pos_pe = SinusoidalPE(d_model, max_len=MAX_TOKENS)
        if self.use_phase:
            self.phase_pe = PhaseSinusoidalPE(d_model)

        # weights for pos & phase
        if self.use_pos:
            self.w_pos = nn.Parameter(torch.tensor(self.alpha_pos)) if weighting in {"learned", "hybrid"} else torch.tensor(self.alpha_pos)
        else:
            self.w_pos = None
        if self.use_phase:
            self.w_phase = nn.Parameter(torch.tensor(self.alpha_phase)) if weighting in {"learned", "hybrid"} else torch.tensor(self.alpha_phase)
        else:
            self.w_phase = None

        # SNR encoding
        if self.use_snr:
            self.snr_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )
            self.w_snr = nn.Parameter(torch.tensor(float(alpha_snr))) if weighting in {"learned", "hybrid"} else torch.tensor(float(alpha_snr))
        else:
            self.snr_mlp = None
            self.w_snr   = None

        # self-attention encoder + head
        self.blocks = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model, dim_ff, heads=heads, p_drop=p_drop
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d_model, out_dim)
        )

    def _pool_phase_to_tokens_by_adaptive(self, phase: torch.Tensor, S: int) -> torch.Tensor:
        if phase.ndim == 3:
            phase = phase.squeeze(-1)
        return F.adaptive_avg_pool1d(phase.unsqueeze(1), output_size=S).squeeze(1)

    def _pool_phase_by_patch(self, phase: torch.Tensor, P: int) -> torch.Tensor:
        if phase.ndim == 3:
            phase = phase.squeeze(-1)
        if phase.size(1) % P != 0:
            phase = F.pad(phase, (0, P - (phase.size(1) % P)))
        chunks = phase.view(phase.size(0), -1, P)  # (B,S,P)
        return chunks.mean(-1) if self.phase_pool == "mean" else chunks[:, :, P//2]

    def forward(self, x: torch.Tensor,
                phase: torch.Tensor | None = None,
                snr: torch.Tensor | None = None) -> torch.Tensor:

        # Accept x: (B,L) or (B,P,L)

        x_orig = x  # keep original shape
        
        # ---- compute phase/snr BEFORE flattening ----
        need_phase = self.use_phase and (phase is None)
        need_snr   = self.use_snr   and (snr   is None)
        
        if (need_phase or need_snr) and (self.phase_provider is not None):
            # make sure provider has correct dataset stats
            self.phase_provider.x_mean = self.x_mean_buf
            self.phase_provider.x_std  = self.x_std_buf
            
            # IMPORTANT: call on original x (B,P,L) if available
            phase_from_provider = self.phase_provider(x_orig)
            if need_phase:
                phase = phase_from_provider
            if need_snr:
                snr = self.phase_provider.last_snr_pred
                if isinstance(snr, torch.Tensor) and snr.ndim == 2:
                    snr = snr.mean(dim=1)
                
        if x.ndim == 3:
            B, Pp, Lp = x.shape
        
            # FIX: if phase is (B,Lp), expand to (B,Pp,Lp) so it matches x after flatten
            if phase is not None and phase.ndim == 2 and phase.size(1) == Lp:
                phase = phase[:, None, :].repeat(1, Pp, 1)  # (B,P,L)
        
            x = x.reshape(B, Pp * Lp)
        
            # keep phase aligned with flattened x
            if phase is not None and phase.ndim == 3:
                phase = phase.reshape(B, Pp * Lp)
        else:
            B = x.size(0)


        # maybe compute φ / SNR via PhaseProvider
        need_phase = self.use_phase and (phase is None)
        need_snr   = self.use_snr   and (snr   is None)

        if (need_phase or need_snr) and (self.phase_provider is not None):
            # sync stats
            if hasattr(self.phase_provider, "x_mean"):
                self.phase_provider.x_mean = self.x_mean_buf
            if hasattr(self.phase_provider, "x_std"):
                self.phase_provider.x_std  = self.x_std_buf

            phase_from_provider = self.phase_provider(x)  # (B,L) or (B,P,L) or (B,P*L)
            if need_phase:
                phase = phase_from_provider

            if need_snr:
                snr = self.phase_provider.last_snr_pred
                # if per-pulsar, average to per-sample
                if isinstance(snr, torch.Tensor) and snr.ndim == 2:
                    snr = snr.mean(dim=1)

        # tokens
        if self.cnn_stem:
            z = self.stem(x)        # (B,S,d)
            S = z.size(1); Ppatch = None
        else:
            z, (_B, _L, Ppatch) = self.embed(x)
            S = z.size(1)

        # index pos encoding
        if self.use_pos:
            pos_term = self.pos_pe.pe[:S].unsqueeze(0).to(z.dtype)
            z = z + (self.w_pos * pos_term if isinstance(self.w_pos, nn.Parameter) else float(self.w_pos) * pos_term)

        # phase encoding
        if self.use_phase:
            if phase is None:
                raise ValueError("use_phasepe=True but no phase provided (and no PhaseProvider)!")
            if phase.ndim == 3:
                phase = phase.reshape(phase.size(0), -1)

            if self.cnn_stem:
                phase_tok = self._pool_phase_to_tokens_by_adaptive(phase, S)  # (B,S)
            else:
                phase_tok = self._pool_phase_by_patch(phase, Ppatch)          # (B,S)

            phi_term = self.phase_pe(phase_tok)  # (B,S,d)
            z = z + (self.w_phase * phi_term if isinstance(self.w_phase, nn.Parameter) else float(self.w_phase) * phi_term)

        # snr encoding
        if self.use_snr:
            if snr is None:
                raise ValueError("use_snrenc=True but no snr provided (and no PhaseProvider)!")
            snr_val = snr.to(z.device, z.dtype).view(B, 1)
            snr_log10 = torch.log10(torch.clamp(snr_val, min=1e-3))
            snr_emb = self.snr_mlp(snr_log10)                  # (B,d_model)
            snr_tok = snr_emb.unsqueeze(1).expand(-1, S, -1)    # (B,S,d_model)
            z = z + (self.w_snr * snr_tok if isinstance(self.w_snr, nn.Parameter) else float(self.w_snr) * snr_tok)

        # self-attention encoder + head
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).mean(1)
        return self.head(z)




class HierarchicalPTAEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_pulsars: int,
        out_dim: int,
        *,
        patch: int = 20,
        d_model: int = 128,
        temporal_depth: int = 2,
        cross_depth: int = 2,
        dim_ff: int = 256,
        heads: int = 8,
        p_drop: float = 0.1,
        use_posenc: bool = True,
        use_phasepe: bool = False,
        phase_pool: str = "mean",
        weighting: str = "learned",
        alpha_pos: float = 1.0,
        alpha_phase: float = 1.0,
        phase_provider = None,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_pulsars = n_pulsars
        self.out_dim = out_dim
        self.use_phasepe = bool(use_phasepe)
        self.phase_provider = phase_provider

        self.register_buffer(
            "x_mean_buf",
            torch.as_tensor(0.0) if x_mean is None else x_mean.clone().detach()
        )
        self.register_buffer(
            "x_std_buf",
            torch.as_tensor(1.0) if x_std is None else x_std.clone().detach()
        )

        self.temporal = PerPulsarTemporalEncoder(
            seq_len=seq_len,
            d_model=d_model,
            patch=patch,
            depth=temporal_depth,
            dim_ff=dim_ff,
            heads=heads,
            p_drop=p_drop,
            use_posenc=use_posenc,
            use_phasepe=use_phasepe,
            phase_pool=phase_pool,
            weighting=weighting,
            alpha_pos=alpha_pos,
            alpha_phase=alpha_phase,
        )

        self.cross = CrossPulsarEncoder(
            n_pulsars=n_pulsars,
            d_model=d_model,
            depth=cross_depth,
            dim_ff=dim_ff,
            heads=heads,
            p_drop=p_drop,
            use_pulsar_posenc=True,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, out_dim),
        )

    def _maybe_get_phase(self, x: torch.Tensor, phase: torch.Tensor | None):
        """
        x:     (B,P,L)
        phase: None or (B,P,L) or (B,L)
        returns phase as (B,P,L) if phase encoding is enabled
        """
        if not self.use_phasepe:
            return None

        if phase is not None:
            if phase.ndim == 2:
                # (B,L) -> (B,P,L)
                B, L = phase.shape
                if L != self.seq_len:
                    raise ValueError(f"Expected phase length {self.seq_len}, got {L}")
                phase = phase[:, None, :].expand(B, self.n_pulsars, L)
            elif phase.ndim == 3:
                pass
            else:
                raise ValueError(f"Unsupported phase shape {tuple(phase.shape)}")
            return phase

        if self.phase_provider is not None:
            # sync stats for provider
            if hasattr(self.phase_provider, "x_mean"):
                self.phase_provider.x_mean = self.x_mean_buf
            if hasattr(self.phase_provider, "x_std"):
                self.phase_provider.x_std = self.x_std_buf

            phase_pred = self.phase_provider(x)

            # provider may return (B,L) or (B,P,L)
            if phase_pred.ndim == 2:
                B, L = phase_pred.shape
                phase_pred = phase_pred[:, None, :].expand(B, self.n_pulsars, L)
            elif phase_pred.ndim != 3:
                raise ValueError(f"Unsupported predicted phase shape {tuple(phase_pred.shape)}")

            return phase_pred

        raise ValueError("use_phasepe=True but neither phase nor phase_provider was given")

    def forward(self, x: torch.Tensor, phase: torch.Tensor | None = None, snr: torch.Tensor | None = None):
        """
        x: (B,P,L)
        phase: optional (B,P,L) or (B,L)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,P,L), got {tuple(x.shape)}")

        B, P, L = x.shape
        if P != self.n_pulsars:
            raise ValueError(f"Expected P={self.n_pulsars}, got {P}")
        if L != self.seq_len:
            raise ValueError(f"Expected L={self.seq_len}, got {L}")

        phase_eff = self._maybe_get_phase(x, phase) if self.use_phasepe else None

        x_bp = x.reshape(B * P, L)   # (B*P, L)

        if phase_eff is not None:
            phase_bp = phase_eff.reshape(B * P, L)   # (B*P, L)
        else:
            phase_bp = None

        z_bp = self.temporal(x_bp, phase_bpL=phase_bp)   # (B*P, d)
        z = z_bp.view(B, P, -1)                          # (B,P,d)
        h = self.cross(z)                                # (B,d)
        return self.head(h)                              # (B,out_dim)

