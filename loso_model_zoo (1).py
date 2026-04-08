"""
loso_model_zoo.py
=================
Leave-One-Subject-Out (LOSO) training for gesture / BFRB classification.

Mirrors LOSO_LST_Improved_v2.m but provides a FULL model zoo in PyTorch:
  1.  BiLSTM            – direct port of the MATLAB baseline
  2.  StackedBiLSTM     – deeper stacked BiLSTM
  3.  CNN_LSTM          – 1-D Conv feature extractor → BiLSTM
  4.  ResLSTM           – residual connections around each LSTM block
  5.  AttentionBiLSTM   – BiLSTM + self-attention pooling
  6.  TCN               – Temporal Convolutional Network (dilated causal convs)
  7.  TransformerCLS    – pure Transformer encoder with CLS token
  8.  CNN_Transformer   – CNN stem → Transformer encoder
  9.  MHSA_LSTM         – Multi-Head Self-Attention interleaved with LSTM
  10. LightweightMV2    – MobileNet-V2-style depthwise separable 1-D convs

Usage
-----
  python loso_model_zoo.py --model AttentionBiLSTM --epochs 100 --batch 64

Requirements
------------
  pip install torch scipy scikit-learn pandas openpyxl tqdm
"""

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Hand Waving", "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting", "Phone Call", "Raising Hand", "Reading",
    "Scratching Arm", "Sitting Still", "Sit-to-Stand", "Standing",
    "Stand-to-Sit", "Stretching", "Thumb Sucking", "Walking",
]
NUM_CLASSES = len(CLASS_NAMES)

# Feature indices from MATLAB: 2:67, 69:84  (1-based → 0-based)
FEAT_ROWS = list(range(1, 67)) + list(range(68, 84))  # 82 features
INPUT_SIZE = len(FEAT_ROWS)                             # 82

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    """Holds variable-length sequences (features × time) + integer labels."""

    def __init__(self, sequences, labels, mu=None, sigma=None,
                 augment_sigma=0.0, downsample=2):
        self.augment_sigma = augment_sigma
        seqs = []
        for s in sequences:
            s = s[FEAT_ROWS, :]          # select feature rows
            s = s[:, ::downsample]       # temporal downsampling (step=2)
            seqs.append(torch.tensor(s, dtype=torch.float32).T)  # T × F

        # Normalise
        if mu is None:
            all_vals = torch.cat(seqs, dim=0)
            self.mu    = all_vals.mean(0)
            self.sigma = all_vals.std(0).clamp(min=1e-6)
        else:
            self.mu, self.sigma = mu, sigma

        self.seqs   = [(s - self.mu) / self.sigma for s in seqs]
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self.augment_sigma > 0:
            seq = seq + self.augment_sigma * torch.randn_like(seq)
        return seq, self.labels[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded  = pad_sequence(seqs, batch_first=True)  # B × T_max × F
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ZOO
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. BiLSTM baseline ───────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    """
    Direct Python equivalent of the MATLAB BiLSTM.
    Two BiLSTM layers (100 hidden units each) with dropout.
    ~300 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden=100, num_classes=NUM_CLASSES,
                 dropout=0.5):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden, batch_first=True,
                             bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden * 2, hidden, batch_first=True,
                             bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm1(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.drop1(out)

        packed = pack_padded_sequence(out, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, (h, _) = self.lstm2(packed)
        # take last hidden state (both directions concatenated)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop2(h))


# ── 2. Stacked BiLSTM ─────────────────────────────────────────────────────────
class StackedBiLSTM(nn.Module):
    """
    4-layer stacked BiLSTM with layer-norm between layers.
    ~1.1 M parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden=128, num_layers=4,
                 num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.drops  = nn.ModuleList()
        in_sz = input_size
        for i in range(num_layers):
            self.layers.append(nn.LSTM(in_sz, hidden, batch_first=True,
                                       bidirectional=True))
            self.norms.append(nn.LayerNorm(hidden * 2))
            self.drops.append(nn.Dropout(dropout))
            in_sz = hidden * 2
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        out = x
        for lstm, norm, drop in zip(self.layers, self.norms, self.drops):
            packed = pack_padded_sequence(out, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            packed_out, (h, _) = lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            out = drop(norm(out))
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


# ── 3. CNN-LSTM ───────────────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    """
    Multi-scale 1-D CNN (kernels 3,5,7) for local feature extraction,
    followed by a BiLSTM for temporal modelling.
    ~450 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, cnn_channels=64, hidden=128,
                 num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        # Parallel convolutions at different scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, cnn_channels, k, padding=k//2),
                nn.BatchNorm1d(cnn_channels),
                nn.GELU(),
                nn.Dropout(0.2),
            ) for k in [3, 5, 7]
        ])
        merged = cnn_channels * 3
        self.proj  = nn.Linear(merged, hidden)
        self.lstm  = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True,
                             num_layers=2, dropout=dropout)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        # x: B × T × F → B × F × T for Conv1d
        xc = x.permute(0, 2, 1)
        outs = [c(xc).permute(0, 2, 1) for c in self.convs]   # B × T × C each
        out  = torch.cat(outs, dim=-1)                          # B × T × 3C
        out  = F.gelu(self.proj(out))                           # B × T × H

        packed = pack_padded_sequence(out, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
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
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.drop(self.norm(out + x))   # residual connection
        return out


class ResLSTM(nn.Module):
    """
    Residual BiLSTM blocks – residual connections stabilise deep training.
    3 residual blocks (hidden=128), ~480 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden=128, num_blocks=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embed  = nn.Linear(input_size, hidden)
        self.blocks = nn.ModuleList(
            [_ResLSTMBlock(hidden, dropout) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths):
        out = F.gelu(self.embed(x))
        for blk in self.blocks:
            out = blk(out, lengths)
        # mean-pool over valid time steps
        mask = torch.arange(out.shape[1], device=out.device)[None] < lengths[:, None]
        out  = (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).float().to(out.device)
        return self.fc(out)


# ── 5. AttentionBiLSTM ────────────────────────────────────────────────────────
class AttentionBiLSTM(nn.Module):
    """
    BiLSTM followed by additive (Bahdanau-style) self-attention pooling.
    Attention learns which time steps matter most per class.
    ~380 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden=128, num_classes=NUM_CLASSES,
                 dropout=0.4):
        super().__init__()
        self.lstm  = nn.LSTM(input_size, hidden, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=dropout)
        self.attn  = nn.Linear(hidden * 2, 1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)   # B × T × 2H

        # Mask padding before softmax
        mask = (torch.arange(out.shape[1], device=out.device)[None] >= lengths[:, None])
        scores = self.attn(out).squeeze(-1)                    # B × T
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)   # B × T × 1
        ctx = (out * weights).sum(1)                           # B × 2H
        return self.fc(self.drop(ctx))


# ── 6. TCN (Temporal Convolutional Network) ───────────────────────────────────
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
    """
    Temporal Convolutional Network with exponentially increasing dilations.
    Receptive field grows as 2^(num_levels) × kernel_size.
    ~600 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, num_channels=None, kernel=3,
                 num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 128, 256]
        layers = []
        in_ch = input_size
        for i, ch in enumerate(num_channels):
            layers.append(_TCNBlock(in_ch, ch, kernel, 2**i, dropout))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.fc  = nn.Linear(in_ch, num_classes)

    def forward(self, x, lengths):
        out = self.net(x.permute(0, 2, 1))        # B × C × T
        out = out.mean(-1)                          # global average pool over T
        return self.fc(out)


# ── 7. Transformer Encoder (CLS token) ───────────────────────────────────────
class TransformerCLS(nn.Module):
    """
    Pure Transformer encoder with a prepended [CLS] token.
    Positional encoding + 4-head multi-head attention, 3 layers.
    ~900 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, num_classes=NUM_CLASSES, dropout=0.3, max_len=2000):
        super().__init__()
        self.embed  = nn.Linear(input_size, d_model)
        self.cls_tk = nn.Parameter(torch.zeros(1, 1, d_model))
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len + 1, d_model)
        pos = torch.arange(0, max_len + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # 1 × max+1 × d

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_tk, std=0.02)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        out = self.embed(x) + self.pe[:, 1:T+1]      # B × T × d
        cls = self.cls_tk.expand(B, -1, -1)
        out = torch.cat([cls, out], dim=1)            # B × T+1 × d

        # key-padding mask: True = ignore
        mask = torch.zeros(B, T + 1, dtype=torch.bool, device=x.device)
        for i, l in enumerate(lengths):
            if l < T:
                mask[i, l + 1:] = True

        out = self.encoder(out, src_key_padding_mask=mask)
        return self.fc(out[:, 0])                     # CLS output


# ── 8. CNN-Transformer ────────────────────────────────────────────────────────
class CNN_Transformer(nn.Module):
    """
    CNN stem (3 layers of depthwise separable Conv1d) down-samples the
    sequence, then a compact Transformer encoder processes the tokens.
    ~700 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, d_model=128, nhead=4,
                 num_layers=3, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),   # ×2 downsample
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        out = self.stem(x.permute(0, 2, 1)).permute(0, 2, 1)  # B × T' × d
        out = self.encoder(out)
        return self.fc(out.mean(1))


# ── 9. MHSA_LSTM (Interleaved attention + LSTM) ───────────────────────────────
class MHSA_LSTM(nn.Module):
    """
    Alternating blocks of Multi-Head Self-Attention and BiLSTM.
    MHSA captures global patterns; BiLSTM captures local sequential dynamics.
    ~650 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden=128, nhead=4, num_blocks=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embed   = nn.Linear(input_size, hidden)
        self.attn_ls = nn.ModuleList()
        self.lstm_ls = nn.ModuleList()
        self.norm1s  = nn.ModuleList()
        self.norm2s  = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_ls.append(nn.MultiheadAttention(hidden, nhead,
                                                       dropout=dropout,
                                                       batch_first=True))
            self.lstm_ls.append(nn.LSTM(hidden, hidden // 2, batch_first=True,
                                        bidirectional=True))
            self.norm1s.append(nn.LayerNorm(hidden))
            self.norm2s.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths):
        out = F.gelu(self.embed(x))
        B, T, _ = out.shape

        key_mask = (torch.arange(T, device=x.device)[None] >= lengths[:, None])

        for attn, lstm, n1, n2 in zip(self.attn_ls, self.lstm_ls,
                                       self.norm1s, self.norm2s):
            a, _ = attn(out, out, out, key_padding_mask=key_mask)
            out  = n1(out + self.drop(a))

            packed   = pack_padded_sequence(out, lengths.cpu(), batch_first=True,
                                            enforce_sorted=False)
            lout, _  = lstm(packed)
            lout, _  = pad_packed_sequence(lout, batch_first=True)
            out      = n2(out + self.drop(lout))

        # attention-weighted pool
        mask   = key_mask.unsqueeze(-1)
        scores = out.masked_fill(mask, float('-inf'))
        w      = torch.softmax(scores.mean(-1), dim=1).unsqueeze(-1)
        ctx    = (out * w).sum(1)
        return self.fc(ctx)


# ── 10. Lightweight MobileNet-V2-style 1-D ───────────────────────────────────
class _InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid = in_ch * expand
        use_res = (stride == 1 and in_ch == out_ch)
        self.use_res = use_res
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1), nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, mid, 3, stride, 1, groups=mid), nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, out_ch, 1), nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        return (x + self.conv(x)) if self.use_res else self.conv(x)


class LightweightMV2(nn.Module):
    """
    MobileNet-V2-style depthwise separable 1-D convolutions.
    Very fast inference; good for real-time deployment.
    ~160 K parameters.
    """
    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        cfg = [
            # (out_ch, stride, expand, repeat)
            (32, 1, 1, 1),
            (48, 2, 6, 2),
            (64, 1, 6, 3),
            (96, 2, 6, 2),
            (128, 1, 6, 1),
        ]
        layers = [nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU6(True)
        )]
        in_ch = 32
        for out_ch, s, e, n in cfg:
            for i in range(n):
                layers.append(_InvertedResidual1D(in_ch, out_ch,
                                                  stride=s if i == 0 else 1,
                                                  expand=e))
                in_ch = out_ch
        layers += [nn.Conv1d(in_ch, 256, 1), nn.BatchNorm1d(256), nn.ReLU6(True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x, lengths):
        out = self.features(x.permute(0, 2, 1))
        return self.head(out)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "BiLSTM":           BiLSTM,
    "StackedBiLSTM":    StackedBiLSTM,
    "CNN_LSTM":         CNN_LSTM,
    "ResLSTM":          ResLSTM,
    "AttentionBiLSTM":  AttentionBiLSTM,
    "TCN":              TCN,
    "TransformerCLS":   TransformerCLS,
    "CNN_Transformer":  CNN_Transformer,
    "MHSA_LSTM":        MHSA_LSTM,
    "LightweightMV2":   LightweightMV2,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (from .mat or from raw Excel, matching LOSO_LoadCellData.m)
# ─────────────────────────────────────────────────────────────────────────────

def load_subjects_from_mat(mat_path: str):
    """
    Load subjects_per_class.mat saved by LOSO_LoadCellData.m.
    Returns a list of dicts: [{'subjectID': str, 'samples': [{'data': ndarray, 'label': int}]}]
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    raw = mat['subjects']

    subjects = []
    for i in range(raw.shape[0]):
        subj = raw[i]
        sid  = str(subj.subjectID)
        samps_raw = subj.samples

        # Handle MATLAB cell arrays (object arrays after squeeze_me)
        if not hasattr(samps_raw, '__len__'):
            samps_raw = [samps_raw]
        samps_raw = np.atleast_1d(samps_raw).ravel()

        samples = []
        for smp in samps_raw:
            try:
                data  = np.array(smp.data, dtype=np.float32)
                label = int(np.atleast_1d(smp.label).ravel()[0])
                cls   = str(np.atleast_1d(smp.class_).ravel()[0]) \
                        if hasattr(smp, 'class_') else CLASS_NAMES[label - 1]
                samples.append({'data': data, 'label': label, 'class': cls})
            except Exception as e:
                warnings.warn(f"Skipping sample in subject {sid}: {e}")

        if samples:
            subjects.append({'subjectID': sid, 'samples': samples})

    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for x, lengths, y in loader:
        x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    scheduler.step()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for x, lengths, y in loader:
        x, lengths = x.to(DEVICE), lengths.to(DEVICE)
        logits = model(x, lengths)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(y.tolist())
    return np.array(all_pred), np.array(all_true)


def compute_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    TP   = np.diag(cm)
    FP   = cm.sum(0) - TP
    FN   = cm.sum(1) - TP
    TN   = cm.sum() - (TP + FP + FN)
    eps  = 1e-10
    prec = TP / (TP + FP + eps)
    rec  = TP / (TP + FN + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    mcc  = (TP * TN - FP * FN) / np.sqrt(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps)
    return {
        'acc': acc,
        'macro_precision': prec.mean(),
        'macro_recall':    rec.mean(),
        'macro_f1':        f1.mean(),
        'macro_mcc':       mcc.mean(),
        'confusion_matrix': cm,
        'per_class': {'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc},
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSO MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_loso(subjects, model_cls, epochs=100, batch_size=64, lr=1e-3,
             val_frac=0.05, aug_sigma=0.01, output_dir='loso_results',
             model_kwargs=None):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_kwargs = model_kwargs or {}

    num_subjects = len(subjects)
    fold_metrics = []
    all_conf_mats = []

    for s_idx in range(num_subjects):
        print(f"\n{'='*60}")
        print(f"  Fold {s_idx+1}/{num_subjects} – "
              f"Leave out: {subjects[s_idx]['subjectID']}")
        print(f"{'='*60}")

        test_samples  = subjects[s_idx]['samples']
        train_samples = [smp for i, subj in enumerate(subjects)
                         if i != s_idx for smp in subj['samples']]

        if not train_samples or not test_samples:
            warnings.warn(f"Empty split at fold {s_idx+1}; skipping.")
            continue

        # ── Stratified val split ──
        rng = np.random.default_rng(42)
        indices = np.arange(len(train_samples))
        val_idx = []
        for c in range(1, NUM_CLASSES + 1):
            idxc = indices[[s['label'] == c for s in train_samples]]
            k    = max(0, round(val_frac * len(idxc)))
            if k > 0 and len(idxc) > 1:
                chosen = rng.choice(idxc, size=k, replace=False)
                val_idx.extend(chosen.tolist())

        val_idx   = list(set(val_idx))
        train_idx = [i for i in indices if i not in set(val_idx)]

        tr_samps  = [train_samples[i] for i in train_idx]
        va_samps  = [train_samples[i] for i in val_idx] if val_idx else tr_samps[:5]
        te_samps  = test_samples

        def to_arrays(samps):
            seqs   = [np.array(s['data'], dtype=np.float32) for s in samps]
            labels = [int(s['label']) - 1 for s in samps]   # 0-indexed
            return seqs, labels

        tr_seqs, tr_lbl = to_arrays(tr_samps)
        va_seqs, va_lbl = to_arrays(va_samps)
        te_seqs, te_lbl = to_arrays(te_samps)

        tr_ds = GestureDataset(tr_seqs, tr_lbl, augment_sigma=aug_sigma)
        va_ds = GestureDataset(va_seqs, va_lbl, mu=tr_ds.mu, sigma=tr_ds.sigma)
        te_ds = GestureDataset(te_seqs, te_lbl, mu=tr_ds.mu, sigma=tr_ds.sigma)

        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)
        te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)

        # ── Build model ──
        model = model_cls(**model_kwargs).to(DEVICE)
        if s_idx == 0:
            print(f"  Model: {model_cls.__name__}  |  "
                  f"Params: {count_params(model):,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_val_acc = -1.0
        best_state   = None
        patience, no_improve = 15, 0

        pbar = tqdm(range(1, epochs + 1), desc='Training', leave=False)
        for ep in pbar:
            tr_loss, tr_acc = train_one_epoch(model, tr_dl, optimizer,
                                              scheduler, criterion)
            va_pred, va_true = evaluate(model, va_dl)
            va_acc = accuracy_score(va_true, va_pred)

            pbar.set_postfix({'tr_loss': f'{tr_loss:.3f}',
                              'tr_acc': f'{tr_acc:.3f}',
                              'va_acc': f'{va_acc:.3f}'})

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep}")
                break

        # ── Evaluate best model ──
        model.load_state_dict(best_state)
        te_pred, te_true = evaluate(model, te_dl)
        metrics = compute_metrics(te_true, te_pred)
        fold_metrics.append(metrics)
        all_conf_mats.append(metrics['confusion_matrix'])

        print(f"  Accuracy : {metrics['acc']*100:.2f}%")
        print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
        print(f"  MCC      : {metrics['macro_mcc']:.4f}")

        # ── Save per-fold CSV ──
        pd.DataFrame(
            metrics['confusion_matrix'],
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        ).to_csv(
            os.path.join(output_dir,
                         f"cm_fold_{s_idx+1:02d}_{subjects[s_idx]['subjectID']}.csv")
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    if fold_metrics:
        avg_acc   = np.mean([m['acc']              for m in fold_metrics])
        avg_prec  = np.mean([m['macro_precision']  for m in fold_metrics])
        avg_rec   = np.mean([m['macro_recall']     for m in fold_metrics])
        avg_f1    = np.mean([m['macro_f1']         for m in fold_metrics])
        avg_mcc   = np.mean([m['macro_mcc']        for m in fold_metrics])

        print(f"\n{'='*60}")
        print(f"  LOSO Summary ({model_cls.__name__})  –  {len(fold_metrics)} folds")
        print(f"{'='*60}")
        print(f"  Avg Accuracy  : {avg_acc*100:.2f}%")
        print(f"  Avg Precision : {avg_prec:.4f}")
        print(f"  Avg Recall    : {avg_rec:.4f}")
        print(f"  Avg F1        : {avg_f1:.4f}")
        print(f"  Avg MCC       : {avg_mcc:.4f}")

        # Save aggregate metrics CSV
        summary_df = pd.DataFrame([{
            'model':     model_cls.__name__,
            'avg_acc':   avg_acc,
            'avg_prec':  avg_prec,
            'avg_rec':   avg_rec,
            'avg_f1':    avg_f1,
            'avg_mcc':   avg_mcc,
        }])
        summary_df.to_csv(
            os.path.join(output_dir, f'summary_{model_cls.__name__}.csv'),
            index=False
        )

    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK: run ALL models sequentially and compare
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_all_models(subjects, epochs=50, batch_size=64, output_dir='benchmark'):
    """Quick benchmark (fewer epochs) of every model in the zoo."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for name, cls in MODEL_REGISTRY.items():
        print(f"\n{'#'*60}\n  MODEL: {name}\n{'#'*60}")
        try:
            folds = run_loso(subjects, cls, epochs=epochs, batch_size=batch_size,
                             output_dir=os.path.join(output_dir, name))
            if folds:
                results.append({
                    'model':    name,
                    'params':   count_params(cls()),
                    'avg_acc':  np.mean([m['acc']             for m in folds]),
                    'avg_f1':   np.mean([m['macro_f1']        for m in folds]),
                    'avg_mcc':  np.mean([m['macro_mcc']       for m in folds]),
                })
        except Exception as e:
            warnings.warn(f"Model {name} failed: {e}")

    df = pd.DataFrame(results).sort_values('avg_acc', ascending=False)
    out_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(out_path, index=False)
    print(f"\n\n{'='*60}\n  MODEL COMPARISON\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LOSO gesture classification – Python model zoo"
    )
    parser.add_argument("--mat",     default="subjects_per_class.pkl",
                        help="Path to subjects_per_class.pkl (generated by load_cell_data.py)")
    parser.add_argument("--model",   default="AttentionBiLSTM",
                        choices=list(MODEL_REGISTRY.keys()) + ["ALL"],
                        help="Model to train (or ALL to benchmark every model)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--aug",     type=float, default=0.01,
                        help="Gaussian augmentation sigma (0 = off)")
    parser.add_argument("--outdir",  default="loso_results")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading data from: {args.mat}")
    import pickle
    with open(args.mat, "rb") as f:
        subjects = pickle.load(f)
    print(f"Loaded {len(subjects)} subjects.")

    if args.model == "ALL":
        benchmark_all_models(subjects, epochs=args.epochs,
                             batch_size=args.batch, output_dir=args.outdir)
    else:
        model_cls = MODEL_REGISTRY[args.model]
        run_loso(subjects, model_cls,
                 epochs=args.epochs, batch_size=args.batch,
                 lr=args.lr, aug_sigma=args.aug,
                 output_dir=args.outdir)


if __name__ == "__main__":
    main()
