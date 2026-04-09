"""
transformer_models.py
=====================
Multiple Transformer variants for time-series / sequence classification.

All models share the same interface:
    model = ModelClass(input_dim, num_classes, seq_len, **kwargs)
    logits = model(x, padding_mask=None)
        x            : (B, T, input_dim)  float32
        padding_mask : (B, T)             bool, True = PAD position (matches
                       PyTorch convention for src_key_padding_mask)

The mask is propagated into every attention layer and masked-mean pooling
so padded time steps never leak into the output.

Available variants
------------------
1. VanillaTransformer        – standard encoder-only transformer
2. PatchTransformer          – splits sequence into patches before encoding
3. TransformerWithConvStem   – CNN stem + transformer (ViT-style for 1-D)
4. HierarchicalTransformer   – two-stage coarse-to-fine encoding
5. CrossAttentionTransformer – learnable class tokens + cross-attention readout
6. ConformerTransformer      – Conformer blocks (conv + attention, popular in audio)
7. LightweightLinformer      – linear-complexity attention (Linformer-inspired)

Helper
------
build_model(name, input_dim, num_classes, seq_len, **kwargs)
    Factory function to build any variant by name.

masked_mean(x, padding_mask)
    Shared utility: mean-pool (B,T,D) ignoring padded positions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, padding_mask) -> torch.Tensor:
    """
    Mean-pool (B, T, D) -> (B, D), excluding padded positions.

    Parameters
    ----------
    x            : (B, T, D) float tensor
    padding_mask : (B, T) bool tensor, True = PAD (same as PyTorch convention).
                   If None, plain mean over T is used.
    """
    if padding_mask is None:
        return x.mean(dim=1)
    valid = (~padding_mask).float().unsqueeze(-1)   # (B, T, 1)  1=real, 0=pad
    summed = (x * valid).sum(dim=1)                 # (B, D)
    counts = valid.sum(dim=1).clamp(min=1.0)        # (B, 1)
    return summed / counts


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 1. VanillaTransformer
# ---------------------------------------------------------------------------
class VanillaTransformer(nn.Module):
    """
    Standard Transformer encoder stack.

    Params
    ------
    d_model        : embedding dimension
    nhead          : number of attention heads
    num_layers     : number of encoder layers
    dim_feedforward: FFN hidden size
    dropout        : dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        x = self.pos_enc(self.proj(x))
        # src_key_padding_mask: True = ignore — same convention as our mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(masked_mean(x, padding_mask))
        return self.head(x)


# ---------------------------------------------------------------------------
# 2. PatchTransformer
# ---------------------------------------------------------------------------
class PatchTransformer(nn.Module):
    """
    Splits the time dimension into non-overlapping patches (ViT-style for 1-D).
    The padding mask is projected to patch-level before the encoder.

    Params
    ------
    patch_size    : number of time steps per patch
    d_model, nhead, num_layers, dim_feedforward, dropout : standard params
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        patch_size: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = input_dim * patch_size
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        num_patches = math.ceil(seq_len / patch_size) + 1  # +1 for CLS
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        p = self.patch_size
        pad_len = (p - T % p) % p
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
            if padding_mask is not None:
                extra = x.new_ones(B, pad_len, dtype=torch.bool)
                padding_mask = torch.cat([padding_mask, extra], dim=1)

        x = x.reshape(B, -1, p * C)        # (B, num_patches, p*C)
        x = self.patch_proj(x)              # (B, num_patches, d_model)

        # Patch-level mask: True if ALL steps in the patch are padding
        patch_mask = None
        if padding_mask is not None:
            pm = padding_mask.reshape(B, -1, p)     # (B, num_patches, p)
            patch_mask = pm.all(dim=-1)             # (B, num_patches)
            # CLS token is never masked
            patch_mask = torch.cat(
                [patch_mask.new_zeros(B, 1), patch_mask], dim=1
            )

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # (B, num_patches+1, d_model)
        x = self.dropout(x + self.pos_embed[:, : x.size(1)])
        x = self.encoder(x, src_key_padding_mask=patch_mask)
        x = self.norm(x[:, 0])              # CLS token — always valid
        return self.head(x)


# ---------------------------------------------------------------------------
# 3. TransformerWithConvStem
# ---------------------------------------------------------------------------
class TransformerWithConvStem(nn.Module):
    """
    1-D CNN feature extractor (stem) followed by a Transformer encoder.
    Same-padding keeps T unchanged; mask is resampled if T changes.

    Params
    ------
    conv_channels  : list of channel sizes for Conv stem layers
    kernel_size    : convolution kernel size
    d_model, nhead, num_layers, dim_feedforward, dropout : standard params
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        conv_channels: list = None,
        kernel_size: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128]

        stem = []
        in_ch = input_dim
        for out_ch in conv_channels:
            stem += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.stem = nn.Sequential(*stem)

        self.proj = nn.Linear(in_ch, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, padding_mask=None):
        # Zero padded positions before conv so they don't bleed into real data
        if padding_mask is not None:
            x = x * (~padding_mask).float().unsqueeze(-1)

        x_conv = self.stem(x.permute(0, 2, 1))   # (B, d_stem, T')
        T_out = x_conv.shape[2]
        x = x_conv.permute(0, 2, 1)              # (B, T', d_stem)

        # Resample mask if T changed (same-pad normally keeps T==T')
        stem_mask = None
        if padding_mask is not None:
            if T_out != padding_mask.shape[1]:
                stem_mask = F.interpolate(
                    padding_mask.float().unsqueeze(1),
                    size=T_out, mode="nearest"
                ).squeeze(1).bool()
            else:
                stem_mask = padding_mask

        x = self.pos_enc(self.proj(x))
        x = self.encoder(x, src_key_padding_mask=stem_mask)
        x = self.norm(masked_mean(x, stem_mask))
        return self.head(x)


# ---------------------------------------------------------------------------
# 4. HierarchicalTransformer
# ---------------------------------------------------------------------------
class HierarchicalTransformer(nn.Module):
    """
    Two-stage encoder:
      Stage 1 (local)  – transformer over fixed-size windows
      Stage 2 (global) – transformer over window representations

    A window is globally masked if ALL its time steps are padding.

    Params
    ------
    window_size               : time steps per local window
    d_local                   : embedding dim for the local stage
    d_global                  : embedding dim for the global stage
    nhead_local, nhead_global : attention heads per stage
    layers_local, layers_global : encoder depth per stage
    dropout                   : dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        window_size: int = 20,
        d_local: int = 64,
        d_global: int = 128,
        nhead_local: int = 2,
        nhead_global: int = 4,
        layers_local: int = 1,
        layers_global: int = 2,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.window_size = window_size

        self.local_proj = nn.Linear(input_dim, d_local)
        self.local_pe = PositionalEncoding(d_local, dropout, max_len=window_size + 1)
        local_layer = nn.TransformerEncoderLayer(
            d_local, nhead_local, d_local * 2, dropout, batch_first=True
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, layers_local)
        self.local_norm = nn.LayerNorm(d_local)

        self.global_proj = nn.Linear(d_local, d_global)
        self.global_pe = PositionalEncoding(d_global, dropout)
        global_layer = nn.TransformerEncoderLayer(
            d_global, nhead_global, d_global * 2, dropout, batch_first=True
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, layers_global)
        self.global_norm = nn.LayerNorm(d_global)
        self.head = nn.Linear(d_global, num_classes)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        w = self.window_size
        pad_len = (w - T % w) % w
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
            if padding_mask is not None:
                extra = x.new_ones(B, pad_len, dtype=torch.bool)
                padding_mask = torch.cat([padding_mask, extra], dim=1)

        T_padded = x.shape[1]
        num_win = T_padded // w

        local_step_mask = None
        win_mask = None
        if padding_mask is not None:
            pm = padding_mask.reshape(B, num_win, w)         # (B, num_win, w)
            local_step_mask = pm.reshape(B * num_win, w)     # per-window step mask
            win_mask = pm.all(dim=-1)                        # (B, num_win) all-pad windows

        x = x.reshape(B * num_win, w, C)
        x = self.local_pe(self.local_proj(x))
        x = self.local_encoder(x, src_key_padding_mask=local_step_mask)
        x = masked_mean(x, local_step_mask)                 # (B*num_win, d_local)
        x_win = x.reshape(B, num_win, -1)                   # (B, num_win, d_local)

        x_win = self.global_pe(self.global_proj(x_win))
        x_win = self.global_encoder(x_win, src_key_padding_mask=win_mask)
        x_win = self.global_norm(masked_mean(x_win, win_mask))
        return self.head(x_win)


# ---------------------------------------------------------------------------
# 5. CrossAttentionTransformer
# ---------------------------------------------------------------------------
class CrossAttentionTransformer(nn.Module):
    """
    Learnable class query tokens attend to the encoded sequence via cross-attention.
    padding_mask is passed to both self-attention (encoder) and cross-attention
    (as key_padding_mask) so padded positions never contribute to query outputs.

    Params
    ------
    num_latents  : number of learnable query vectors
    d_model      : encoding dim (self-attention stage)
    d_latent     : latent query dim (cross-attention stage)
    nhead        : heads for self-attention encoder
    num_layers   : depth of self-attention encoder
    nhead_cross  : heads for cross-attention readout
    dropout      : dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        num_latents: int = 16,
        d_model: int = 128,
        d_latent: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        nhead_cross: int = 4,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_latent))
        self.cross_attn = nn.MultiheadAttention(
            d_latent, nhead_cross, dropout=dropout, batch_first=True,
            kdim=d_model, vdim=d_model
        )
        self.cross_norm = nn.LayerNorm(d_latent)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_latents * d_latent, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, padding_mask=None):
        B = x.size(0)
        x = self.pos_enc(self.proj(x))
        mem = self.encoder(x, src_key_padding_mask=padding_mask)    # (B, T, d_model)
        q = self.latents.expand(B, -1, -1)                          # (B, num_latents, d_latent)
        # key_padding_mask masks which memory positions the queries should ignore
        out, _ = self.cross_attn(q, mem, mem, key_padding_mask=padding_mask)
        out = self.cross_norm(out)
        return self.head(out)


# ---------------------------------------------------------------------------
# 6. ConformerTransformer
# ---------------------------------------------------------------------------
class ConformerBlock(nn.Module):
    """One Conformer block: FF -> MHSA -> Conv -> FF with residuals."""

    def __init__(self, d_model: int, nhead: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        padding = (kernel_size - 1) // 2
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        x = x + 0.5 * self.ff1(x)
        y = self.norm_attn(x)
        y, _ = self.attn(y, y, y, key_padding_mask=padding_mask)
        x = x + self.attn_drop(y)
        # Zero padded positions before depthwise conv to prevent cross-contamination
        if padding_mask is not None:
            x = x * (~padding_mask).float().unsqueeze(-1)
        z = self.conv_norm(x).permute(0, 2, 1)
        x = x + self.conv_module(z).permute(0, 2, 1)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConformerTransformer(nn.Module):
    """
    Stack of Conformer blocks.

    Params
    ------
    d_model     : embedding dimension
    nhead       : attention heads
    num_layers  : number of Conformer blocks
    kernel_size : depthwise conv kernel size
    dropout     : dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        kernel_size: int = 15,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, nhead, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        x = self.pos_enc(self.proj(x))
        for blk in self.blocks:
            x = blk(x, padding_mask=padding_mask)
        x = self.norm(masked_mean(x, padding_mask))
        return self.head(x)


# ---------------------------------------------------------------------------
# 7. LightweightLinformer
# ---------------------------------------------------------------------------
class LinformerAttention(nn.Module):
    """
    Linear-complexity attention via low-rank K/V projections.
    Padded positions are zeroed before projection so they cannot
    contribute to the low-rank key/value representations.
    """

    def __init__(self, d_model: int, nhead: int, seq_len: int, k: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.E = nn.Linear(seq_len, k, bias=False)   # K projection T -> k
        self.F = nn.Linear(seq_len, k, bias=False)   # V projection T -> k
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _project(self, t: torch.Tensor, T: int):
        """Project (B,H,D,T)->(B,H,D,k), handling T != stored seq_len via interpolation."""
        seq_len_ref = self.E.weight.shape[1]
        if T == seq_len_ref:
            return self.E(t), self.F(t)
        E_w = F.interpolate(
            self.E.weight.unsqueeze(0), size=T, mode="linear", align_corners=False
        ).squeeze(0)
        F_w = F.interpolate(
            self.F.weight.unsqueeze(0), size=T, mode="linear", align_corners=False
        ).squeeze(0)
        BHD = t.shape[0] * t.shape[1] * t.shape[2]
        k_p = (E_w @ t.reshape(BHD, T).T).T.reshape(*t.shape[:3], -1)
        v_p = (F_w @ t.reshape(BHD, T).T).T.reshape(*t.shape[:3], -1)
        return k_p, v_p

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        H, D = self.nhead, self.d_head

        # Zero padded positions so they don't affect K/V low-rank projections
        if padding_mask is not None:
            x_kv = x * (~padding_mask).float().unsqueeze(-1)
        else:
            x_kv = x

        q = self.q(x).reshape(B, T, H, D).permute(0, 2, 1, 3)      # (B,H,T,D)
        k = self.k(x_kv).reshape(B, T, H, D).permute(0, 2, 3, 1)   # (B,H,D,T)
        v = self.v(x_kv).reshape(B, T, H, D).permute(0, 2, 3, 1)   # (B,H,D,T)

        k_proj, v_proj = self._project(k, T)    # (B,H,D,k) each

        attn = torch.softmax(q @ k_proj * self.scale, dim=-1)        # (B,H,T,k)
        attn = self.drop(attn)
        out = attn @ v_proj.permute(0, 1, 3, 2)                      # (B,H,T,D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class LightweightLinformer(nn.Module):
    """
    Transformer with linear-complexity Linformer attention.

    Params
    ------
    d_model    : embedding dim
    nhead      : attention heads
    num_layers : number of Linformer blocks
    k          : low-rank projection dimension
    dropout    : dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        k: int = 32,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn":  LinformerAttention(d_model, nhead, seq_len, k, dropout),
                "norm1": nn.LayerNorm(d_model),
                "ff":    nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
                ),
                "norm2": nn.LayerNorm(d_model),
                "drop":  nn.Dropout(dropout),
            }))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        x = self.pos_enc(self.proj(x))
        for layer in self.layers:
            x = layer["norm1"](x + layer["drop"](layer["attn"](x, padding_mask)))
            x = layer["norm2"](x + layer["drop"](layer["ff"](x)))
        x = self.final_norm(masked_mean(x, padding_mask))
        return self.head(x)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "vanilla":         VanillaTransformer,
    "patch":           PatchTransformer,
    "conv_stem":       TransformerWithConvStem,
    "hierarchical":    HierarchicalTransformer,
    "cross_attention": CrossAttentionTransformer,
    "conformer":       ConformerTransformer,
    "linformer":       LightweightLinformer,
}


def build_model(name: str, input_dim: int, num_classes: int, seq_len: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_dim, num_classes, seq_len, **kwargs)


# ---------------------------------------------------------------------------
# Quick sanity check: with AND without mask
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, C = 4, 65, 83
    dummy = torch.randn(B, T, C)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, -10:] = True    # last 10 positions are padding

    for label, m_arg in [("no mask", None), ("with mask", mask)]:
        print(f"\n{label}:")
        for name, cls in MODEL_REGISTRY.items():
            try:
                m = cls(C, 20, T)
                out = m(dummy, padding_mask=m_arg)
                print(f"  [OK] {name:25s} output={tuple(out.shape)}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
