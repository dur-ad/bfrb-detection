"""
loso_model_zoo.py
====================
Leave-One-Subject-Out (LOSO) training for gesture / BFRB classification.

Data input : subjects_per_class.pkl  (produced by load_cell_data.py)
             Each sample: {"data": ndarray (features × time), "label": int, "class": str}

Model zoo  :
  1.  BiLSTM
  2.  StackedBiLSTM
  3.  CNN_LSTM
  4.  ResLSTM
  5.  AttentionBiLSTM
  6.  TCN
  7.  TransformerCLS
  8.  CNN_Transformer
  9.  MHSA_LSTM
  10. LightweightMV2
  11. LSTM
  12. GRU
  13. BiGRU

Usage
-----
  python loso_model_zoo_v2.py --model AttentionBiLSTM --epochs 100 --batch_size 32
  python loso_model_zoo_v2.py --model all --epochs 50
"""

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 20
DOWNSAMPLE  = 2        # take every 2nd time step
AUG_SIGMA   = 0.01     # Gaussian noise std for training augmentation
VAL_FRAC    = 0.05     # stratified validation fraction

CLASS_NAMES = [
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Hand Waving", "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting", "Phone Call", "Raising Hand", "Reading",
    "Scratching Arm", "Sitting Still", "Sit-to-Stand", "Standing",
    "Stand-to-Sit", "Stretching", "Thumb Sucking", "Walking",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Variable-length multivariate time-series dataset."""

    def __init__(self, sequences: list, labels: list):
        self.sequences = sequences   # list of (T_i, C) float32 tensors
        self.labels    = labels      # list of int (0-indexed)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """Pad sequences to max length; return (padded, labels, lengths, padding_mask)."""
    seqs, labels = zip(*batch)
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len  = int(lengths.max())
    C        = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, C, dtype=torch.float32)
    # padding_mask: True = padding position (used by Transformer models)
    pad_mask = torch.ones(len(seqs), max_len, dtype=torch.bool)

    for i, (s, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L]    = s
        pad_mask[i, :L]  = False   # False = real data

    return padded, torch.tensor(labels, dtype=torch.long), lengths, pad_mask


def extract_sequence(sample: dict) -> torch.Tensor | None:
    """
    Extract a downsampled sequence from a raw sample dict using ALL feature rows.
    Returns (T', C) float32 tensor or None if invalid / contains NaNs.
    """
    data = sample["data"]   # (features, time)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        return None
    if np.isnan(data).any():
        return None
    seq = data[:, ::DOWNSAMPLE].T.astype(np.float32)   # (T', C)
    return torch.from_numpy(seq)


def compute_normalization(sequences: list):
    """Per-feature mean and std from a list of (T, C) tensors."""
    all_data = np.concatenate([s.numpy() for s in sequences], axis=0)
    mu    = all_data.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = all_data.std(axis=0,  keepdims=True).astype(np.float32)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def apply_norm_aug(seqs: list, mu, sigma, augment: bool = False) -> list:
    out = []
    for s in seqs:
        n = (s.numpy() - mu) / sigma
        if augment:
            n = n + np.random.randn(*n.shape).astype(np.float32) * AUG_SIGMA
        out.append(torch.from_numpy(n))
    return out


def stratified_val_split(labels: list, val_frac: float, seed: int = 1):
    rng        = np.random.default_rng(seed)
    all_idx    = np.arange(len(labels))
    labels_arr = np.array(labels)
    val_idx    = []
    for c in range(NUM_CLASSES):
        idx_c = all_idx[labels_arr == c]
        k = max(1, round(val_frac * len(idx_c))) if len(idx_c) > 1 else 0
        if k > 0:
            val_idx.extend(rng.choice(idx_c, k, replace=False).tolist())
    val_idx   = list(set(val_idx))
    train_idx = [i for i in all_idx if i not in set(val_idx)]
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ZOO
# All models accept forward(x, lengths, padding_mask) for a unified interface.
# lengths     : (B,) actual sequence lengths — used by RNN models
# padding_mask: (B, T) bool, True=pad — used by Transformer/Attention models
# ─────────────────────────────────────────────────────────────────────────────

# ── 0a. LSTM (basic unidirectional) ──────────────────────────────────────────
class LSTM(nn.Module):
    def __init__(self, input_size, hidden=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        packed    = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # h[-1] = last layer's hidden state
        return self.fc(self.drop(h[-1]))


# ── 0b. GRU (basic unidirectional) ───────────────────────────────────────────
class GRU(nn.Module):
    def __init__(self, input_size, hidden=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        packed  = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h    = self.gru(packed)
        return self.fc(self.drop(h[-1]))


# ── 0c. BiGRU (bidirectional GRU) ────────────────────────────────────────────
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        packed  = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h    = self.gru(packed)
        # h[-2] = last layer forward, h[-1] = last layer backward
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(h))

# ── 1. BiLSTM ─────────────────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden=100, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden * 2, hidden, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _      = self.lstm1(packed)
        out, _      = pad_packed_sequence(out, batch_first=True)
        out         = self.drop1(out)
        packed      = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _)   = self.lstm2(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop2(h))


# ── 2. StackedBiLSTM ──────────────────────────────────────────────────────────
class StackedBiLSTM(nn.Module):
    def __init__(self, input_size, hidden=128, num_layers=4,
                 num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.drops  = nn.ModuleList()
        in_sz = input_size
        for _ in range(num_layers):
            self.layers.append(nn.LSTM(in_sz, hidden, batch_first=True, bidirectional=True))
            self.norms.append(nn.LayerNorm(hidden * 2))
            self.drops.append(nn.Dropout(dropout))
            in_sz = hidden * 2
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        out = x
        h_last = None
        for lstm, norm, drop in zip(self.layers, self.norms, self.drops):
            packed          = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h, _) = lstm(packed)
            out, _          = pad_packed_sequence(packed_out, batch_first=True)
            out             = drop(norm(out))
            h_last          = h
        h = torch.cat([h_last[-2], h_last[-1]], dim=-1)
        return self.fc(h)


# ── 3. CNN_LSTM ───────────────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, cnn_channels=64, hidden=128,
                 num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, cnn_channels, k, padding=k // 2),
                nn.BatchNorm1d(cnn_channels), nn.GELU(), nn.Dropout(0.2),
            ) for k in [3, 5, 7]
        ])
        merged    = cnn_channels * 3
        self.proj = nn.Linear(merged, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        xc   = x.permute(0, 2, 1)
        outs = [c(xc).permute(0, 2, 1) for c in self.convs]
        out  = torch.cat(outs, dim=-1)
        out  = F.gelu(self.proj(out))
        packed      = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _)   = self.lstm(packed)
        h = self.drop(torch.cat([h[-2], h[-1]], dim=-1))
        return self.fc(h)


# ── 4. ResLSTM ────────────────────────────────────────────────────────────────
class _ResLSTMBlock(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(size, size // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _      = self.lstm(packed)
        out, _      = pad_packed_sequence(out, batch_first=True)
        # residual: pad out to match x length if needed
        if out.shape[1] < x.shape[1]:
            pad = torch.zeros(x.shape[0], x.shape[1] - out.shape[1], x.shape[2], device=x.device)
            out = torch.cat([out, pad], dim=1)
        return self.drop(self.norm(out + x))


class ResLSTM(nn.Module):
    def __init__(self, input_size, hidden=128, num_blocks=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embed  = nn.Linear(input_size, hidden)
        self.blocks = nn.ModuleList([_ResLSTMBlock(hidden, dropout) for _ in range(num_blocks)])
        self.fc     = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        out  = F.gelu(self.embed(x))
        for blk in self.blocks:
            out = blk(out, lengths)
        # mean-pool over valid time steps only
        mask = (torch.arange(out.shape[1], device=out.device)[None] < lengths[:, None]).unsqueeze(-1).float()
        out  = (out * mask).sum(1) / lengths.unsqueeze(-1).float().to(out.device)
        return self.fc(out)


# ── 5. AttentionBiLSTM ────────────────────────────────────────────────────────
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden=128, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=dropout)
        self.attn = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _      = self.lstm(packed)
        out, _      = pad_packed_sequence(out, batch_first=True)
        # mask padding before softmax
        mask   = torch.arange(out.shape[1], device=out.device)[None] >= lengths[:, None]
        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (out * weights).sum(1)
        return self.fc(self.drop(ctx))


# ── 6. TCN ────────────────────────────────────────────────────────────────────
class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.gelu(self.net(x) + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, input_size, num_channels=None, kernel=3,
                 num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 128, 256]
        layers, in_ch = [], input_size
        for i, ch in enumerate(num_channels):
            layers.append(_TCNBlock(in_ch, ch, kernel, 2 ** i, dropout))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.fc  = nn.Linear(in_ch, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        out = self.net(x.permute(0, 2, 1))   # B × C × T
        out = out.mean(-1)                     # global avg pool
        return self.fc(out)


# ── 7. TransformerCLS ─────────────────────────────────────────────────────────
class TransformerCLS(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, num_classes=NUM_CLASSES, dropout=0.3, max_len=4000):
        super().__init__()
        self.embed  = nn.Linear(input_size, d_model)
        self.cls_tk = nn.Parameter(torch.zeros(1, 1, d_model))
        pe  = torch.zeros(max_len + 1, d_model)
        pos = torch.arange(0, max_len + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_tk, std=0.02)

    def forward(self, x, lengths, padding_mask=None):
        B, T, _ = x.shape
        out = self.embed(x) + self.pe[:, 1:T + 1]
        cls = self.cls_tk.expand(B, -1, -1)
        out = torch.cat([cls, out], dim=1)   # prepend CLS token
        # extend padding_mask to cover CLS token
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            full_mask = None
        out = self.encoder(out, src_key_padding_mask=full_mask)
        return self.fc(out[:, 0])   # CLS output


# ── 8. CNN_Transformer ────────────────────────────────────────────────────────
class CNN_Transformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        # CNN stem halves sequence length — recompute mask accordingly
        out = self.stem(x.permute(0, 2, 1)).permute(0, 2, 1)   # B × T' × d
        if padding_mask is not None:
            # downsample mask to match stem output length
            stem_len = out.shape[1]
            stem_mask = padding_mask[:, ::2][:, :stem_len]
            # ensure same length (rounding)
            if stem_mask.shape[1] < stem_len:
                pad = torch.ones(out.shape[0], stem_len - stem_mask.shape[1],
                                 dtype=torch.bool, device=out.device)
                stem_mask = torch.cat([stem_mask, pad], dim=1)
        else:
            stem_mask = None
        out = self.encoder(out, src_key_padding_mask=stem_mask)
        return self.fc(out.mean(1))


# ── 9. MHSA_LSTM ─────────────────────────────────────────────────────────────
class MHSA_LSTM(nn.Module):
    def __init__(self, input_size, hidden=128, nhead=4, num_blocks=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embed   = nn.Linear(input_size, hidden)
        self.attn_ls = nn.ModuleList()
        self.lstm_ls = nn.ModuleList()
        self.norm1s  = nn.ModuleList()
        self.norm2s  = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_ls.append(nn.MultiheadAttention(hidden, nhead,
                                                       dropout=dropout, batch_first=True))
            self.lstm_ls.append(nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True))
            self.norm1s.append(nn.LayerNorm(hidden))
            self.norm2s.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths, padding_mask=None):
        out      = F.gelu(self.embed(x))
        B, T, _  = out.shape
        key_mask = padding_mask if padding_mask is not None else \
                   (torch.arange(T, device=x.device)[None] >= lengths[:, None])

        for attn, lstm, n1, n2 in zip(self.attn_ls, self.lstm_ls, self.norm1s, self.norm2s):
            a, _  = attn(out, out, out, key_padding_mask=key_mask)
            out   = n1(out + self.drop(a))
            packed        = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lout, _       = lstm(packed)
            lout, _       = pad_packed_sequence(lout, batch_first=True)
            if lout.shape[1] < out.shape[1]:
                pad  = torch.zeros(B, out.shape[1] - lout.shape[1], out.shape[2], device=out.device)
                lout = torch.cat([lout, pad], dim=1)
            out = n2(out + self.drop(lout))

        # attention-weighted pool over valid frames
        scores = out.masked_fill(key_mask.unsqueeze(-1), float('-inf'))
        w      = torch.softmax(scores.mean(-1), dim=1).unsqueeze(-1)
        ctx    = (out * w).sum(1)
        return self.fc(ctx)


# ── 10. LightweightMV2 ────────────────────────────────────────────────────────
class _InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid      = in_ch * expand
        use_res  = (stride == 1 and in_ch == out_ch)
        self.use_res = use_res
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1), nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, mid, 3, stride, 1, groups=mid), nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, out_ch, 1), nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        return (x + self.conv(x)) if self.use_res else self.conv(x)


class LightweightMV2(nn.Module):
    def __init__(self, input_size, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        cfg = [(32, 1, 1, 1), (48, 2, 6, 2), (64, 1, 6, 3), (96, 2, 6, 2), (128, 1, 6, 1)]
        layers = [nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU6(True)
        )]
        in_ch = 32
        for out_ch, s, e, n in cfg:
            for i in range(n):
                layers.append(_InvertedResidual1D(in_ch, out_ch, stride=s if i == 0 else 1, expand=e))
                in_ch = out_ch
        layers += [nn.Conv1d(in_ch, 256, 1), nn.BatchNorm1d(256), nn.ReLU6(True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x, lengths, padding_mask=None):
        out = self.features(x.permute(0, 2, 1))
        return self.head(out)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "LSTM":            LSTM,           
    "GRU":             GRU,            
    "BiGRU":           BiGRU,          
    "BiLSTM":          BiLSTM,
    "StackedBiLSTM":   StackedBiLSTM,
    "CNN_LSTM":        CNN_LSTM,
    "ResLSTM":         ResLSTM,
    "AttentionBiLSTM": AttentionBiLSTM,
    "TCN":             TCN,
    "TransformerCLS":  TransformerCLS,
    "CNN_Transformer": CNN_Transformer,
    "MHSA_LSTM":       MHSA_LSTM,
    "LightweightMV2":  LightweightMV2,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y, lengths, pad_mask in loader:
        x, y        = x.to(device), y.to(device)
        lengths     = lengths.to(device)
        pad_mask    = pad_mask.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, lengths, pad_mask)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, lengths, pad_mask)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y, lengths, pad_mask in loader:
        x, y     = x.to(device), y.to(device)
        lengths  = lengths.to(device)
        pad_mask = pad_mask.to(device)
        logits   = model(x, lengths, pad_mask)
        loss     = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def compute_metrics(conf_mat: np.ndarray):
    TP  = np.diag(conf_mat).astype(float)
    FP  = conf_mat.sum(axis=0) - TP
    FN  = conf_mat.sum(axis=1) - TP
    TN  = conf_mat.sum() - (TP + FP + FN)
    eps = 1e-10
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    mcc       = (TP * TN - FP * FN) / np.sqrt(
                (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps)
    return {
        "accuracy":        TP.sum() / conf_mat.sum(),
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "mcc":             mcc,
        "macro_precision": precision.mean(),
        "macro_recall":    recall.mean(),
        "macro_f1":        f1.mean(),
        "macro_mcc":       mcc.mean(),
    }


def save_confusion_matrix(conf_mat, class_names, title, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_cm  = np.divide(conf_mat, row_sums, where=row_sums != 0)
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE BATCH / GPU→CPU FALLBACK  (mirrors script 3)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_adaptive_batch(model, make_train_dl, make_val_dl,
                               make_optimizer, make_scheduler,
                               criterion, args, device_str, start_batch):
    MIN_BATCH = 8
    batch     = start_batch

    while True:
        try:
            device    = torch.device(device_str)
            model     = model.to(device)
            optimizer = make_optimizer(model)
            scheduler = make_scheduler(optimizer)
            train_dl  = make_train_dl(batch)
            val_dl    = make_val_dl(batch)
            scaler    = torch.cuda.amp.GradScaler() if device_str == "cuda" else None

            best_val_acc = 0.0
            best_state   = None
            patience_cnt = 0

            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc = train_one_epoch(
                    model, train_dl, optimizer, criterion, device, scaler)
                vl_loss, vl_acc, _, _ = evaluate(model, val_dl, criterion, device)
                scheduler.step()

                if epoch % args.log_interval == 0 or epoch == args.epochs:
                    print(f"  Epoch {epoch:3d}/{args.epochs} | "
                          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                          f"val loss {vl_loss:.4f} acc {vl_acc:.3f}")

                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                        break

            if best_state:
                model.load_state_dict(best_state)

            used_env = "gpu" if device_str == "cuda" else "cpu"
            return model, device, used_env, batch

        except RuntimeError as e:
            msg    = str(e).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)

            if is_oom and device_str == "cuda" and batch > MIN_BATCH:
                new_batch = max(MIN_BATCH, batch // 2)
                print(f"  [OOM] batch={batch} → retrying with batch={new_batch}")
                torch.cuda.empty_cache()
                batch = new_batch
                continue

            if is_oom and device_str == "cuda":
                print(f"  [OOM] GPU failed at batch={batch}. Falling back to CPU.")
                device_str = "cpu"
                batch = start_batch
                continue

            raise   # non-OOM error


# ─────────────────────────────────────────────────────────────────────────────
# LOSO MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_loso(subjects, model_name, args):
    device_str   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device_str.upper()}")

    num_subjects = len(subjects)
    max_folds    = args.max_folds if args.max_folds > 0 else num_subjects

    # Output dirs
    cm_dir   = Path(args.out_dir) / model_name / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    excel_cm = Path(args.out_dir) / model_name / "LOSO_ConfusionMatrices.xlsx"
    excel_mt = Path(args.out_dir) / model_name / "LOSO_PerClassMetrics.xlsx"
    for xf in (excel_cm, excel_mt):
        if xf.exists():
            xf.unlink()

    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    fold_metrics = []

    for s_idx in range(min(num_subjects, max_folds)):
        subj = subjects[s_idx]
        sid  = subj["subjectID"]
        print(f"\n=== Fold {s_idx+1}/{min(num_subjects, max_folds)} – Leave out: {sid} ===")

        # ── Build pool (train + val) and test sets ──────────────────────────
        pool_seqs, pool_labels = [], []
        for i, other in enumerate(subjects):
            if i == s_idx:
                continue
            for smp in other["samples"]:
                seq = extract_sequence(smp)
                if seq is not None:
                    pool_seqs.append(seq)
                    pool_labels.append(smp["label"] - 1)   # 0-indexed

        test_seqs, test_labels = [], []
        for smp in subj["samples"]:
            seq = extract_sequence(smp)
            if seq is not None:
                test_seqs.append(seq)
                test_labels.append(smp["label"] - 1)

        if not pool_seqs or not test_seqs:
            warnings.warn(f"Fold {s_idx+1} empty — skipping.")
            continue

        # ── Stratified val split ─────────────────────────────────────────────
        train_idx, val_idx = stratified_val_split(pool_labels, VAL_FRAC, seed=s_idx + 1)
        train_seqs_raw = [pool_seqs[i] for i in train_idx]
        train_labels   = [pool_labels[i] for i in train_idx]
        val_seqs_raw   = [pool_seqs[i] for i in val_idx]
        val_labels     = [pool_labels[i] for i in val_idx]

        # ── Normalize (fit on train+val, apply to all) ───────────────────────
        mu, sigma = compute_normalization(train_seqs_raw + val_seqs_raw)

        train_seqs  = apply_norm_aug(train_seqs_raw, mu, sigma, augment=True)
        val_seqs    = apply_norm_aug(val_seqs_raw,   mu, sigma, augment=False)
        test_seqs_n = apply_norm_aug(test_seqs,      mu, sigma, augment=False)

        # ── Datasets & DataLoader factories ─────────────────────────────────
        pin      = device_str == "cuda"
        train_ds = SequenceDataset(train_seqs,  train_labels)
        val_ds   = SequenceDataset(val_seqs,    val_labels)
        test_ds  = SequenceDataset(test_seqs_n, test_labels)

        def make_train_dl(bs):
            return DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin)
        def make_val_dl(bs):
            return DataLoader(val_ds, batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

        # ── Build model (input_size auto-detected from data) ─────────────────
        input_size = train_seqs[0].shape[1]
        model_cls  = MODEL_REGISTRY[model_name]
        model      = model_cls(input_size=input_size, num_classes=NUM_CLASSES)

        if s_idx == 0:
            print(f"  Model     : {model_name}")
            print(f"  Input dim : {input_size}")
            print(f"  Params    : {count_params(model):,}")

        # ── Train with adaptive batch / GPU→CPU fallback ─────────────────────
        def make_optimizer(m):
            return torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        def make_scheduler(opt):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=args.lr * 0.01)

        model, device, used_env, used_batch = train_with_adaptive_batch(
            model, make_train_dl, make_val_dl,
            make_optimizer, make_scheduler,
            criterion, args, device_str, args.batch_size
        )
        print(f"  Trained on {used_env.upper()} | batch_size={used_batch}")

        # ── Test ─────────────────────────────────────────────────────────────
        model.to(device)
        _, test_acc, y_pred, y_true = evaluate(model, test_dl, criterion, device)
        print(f"  Test accuracy: {test_acc * 100:.2f}%")

        # ── Metrics & confusion matrix ────────────────────────────────────────
        conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        metrics  = compute_metrics(conf_mat)

        fold_metrics.append({
            "subject":   sid,
            "accuracy":  metrics["accuracy"],
            "precision": metrics["macro_precision"],
            "recall":    metrics["macro_recall"],
            "f1":        metrics["macro_f1"],
            "mcc":       metrics["macro_mcc"],
        })

        # ── Save confusion matrix PNG ─────────────────────────────────────────
        safe_id  = "".join(c if c.isalnum() else "_" for c in sid)
        cm_path  = str(cm_dir / f"CM_Fold_{s_idx+1:02d}_{safe_id}.png")
        save_confusion_matrix(conf_mat, CLASS_NAMES,
                              title=f"CM – {sid}  [{model_name}]", path=cm_path)

        # ── Save confusion matrix to Excel ────────────────────────────────────
        sheet = f"Subject_{s_idx+1}"
        cm_df = pd.DataFrame(conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        mode  = "a" if excel_cm.exists() else "w"
        with pd.ExcelWriter(excel_cm, engine="openpyxl", mode=mode,
                            if_sheet_exists="replace" if mode == "a" else None) as w:
            cm_df.to_excel(w, sheet_name=sheet)

        per_class = pd.DataFrame({
            "Precision": metrics["precision"],
            "Recall":    metrics["recall"],
            "F1Score":   metrics["f1"],
            "MCC":       metrics["mcc"],
        }, index=CLASS_NAMES)
        mode = "a" if excel_mt.exists() else "w"
        with pd.ExcelWriter(excel_mt, engine="openpyxl", mode=mode,
                            if_sheet_exists="replace" if mode == "a" else None) as w:
            per_class.to_excel(w, sheet_name=sheet)

    # ── Summary ──────────────────────────────────────────────────────────────
    if fold_metrics:
        print(f"\n=== Summary [{model_name}] — {len(fold_metrics)} folds ===")
        for k in ["accuracy", "precision", "recall", "f1", "mcc"]:
            print(f"  Avg {k:10s}: {np.mean([m[k] for m in fold_metrics]):.4f}")
        summary_df   = pd.DataFrame(fold_metrics)
        summary_path = Path(args.out_dir) / model_name / "LOSO_Summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Summary saved → {summary_path}")

    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO model zoo training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",         default="subjects_cleaned_11subjects.pkl")
    p.add_argument("--model",        default="all",
                   choices=list(MODEL_REGISTRY) + ["all"])
    p.add_argument("--out_dir",      default="loso_results")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--log_interval", type=int,   default=10)
    p.add_argument("--max_folds",    type=int,   default=0,
                   help="0 = all subjects")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)

    import pickle
    with open(args.data, "rb") as f:
        subjects = pickle.load(f)
    print(f"Loaded {len(subjects)} subjects from {args.data}")

    models_to_run = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    all_results   = {}
    model_times   = {}

    for model_name in models_to_run:
        print(f"\n{'#'*60}\n  MODEL: {model_name}\n{'#'*60}")
        t0 = time.time()
        try:
            results = run_loso(subjects, model_name, args)
            all_results[model_name] = results
        except Exception as e:
            warnings.warn(f"Model {model_name} failed: {e}")
            all_results[model_name] = []
        model_times[model_name] = (time.time() - t0) / 3600

    # ── Cross-model comparison ────────────────────────────────────────────────
    if len(models_to_run) > 1:
        rows = []
        for mname, folds in all_results.items():
            if not folds:
                continue
            rows.append({
                "model":     mname,
                "params":    count_params(MODEL_REGISTRY[mname](input_size=1)),
                "accuracy":  np.mean([m["accuracy"]  for m in folds]),
                "precision": np.mean([m["precision"] for m in folds]),
                "recall":    np.mean([m["recall"]    for m in folds]),
                "f1":        np.mean([m["f1"]        for m in folds]),
                "mcc":       np.mean([m["mcc"]       for m in folds]),
            })
        comp_df   = pd.DataFrame(rows).sort_values("f1", ascending=False)
        comp_path = Path(args.out_dir) / "model_comparison.csv"
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(comp_path, index=False)
        print(f"\n{'='*60}\n  MODEL COMPARISON\n{'='*60}")
        print(comp_df.to_string(index=False))
        print(f"\nSaved → {comp_path}")

    print(f"\n{'='*60}\n  TRAINING TIME SUMMARY\n{'='*60}")
    for m, t in model_times.items():
        print(f"  {m:20s}: {t:.2f} hours")
    print(f"  Total: {sum(model_times.values()):.2f} hours")


if __name__ == "__main__":
    main()
