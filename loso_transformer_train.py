"""
loso_transformer_train.py
=========================
Python / PyTorch equivalent of LOSO_LST_Improved_v2.m

Key differences from the MATLAB version:
  - BiLSTM replaced by any of 7 Transformer variants (see transformer_models.py)
  - Variable-length sequences handled with padding + attention masks
  - Stratified 5% validation split
  - Per-fold confusion matrices saved as PNG
  - Per-class metrics + overall summary written to Excel
  - GPU → CPU adaptive fallback
  - Run one model or sweep all variants

Usage examples
--------------
# Single model (conformer, default hyper-params)
python loso_transformer_train.py --model conformer

# Sweep all 7 variants
python loso_transformer_train.py --model all

# Custom hyper-params for vanilla
python loso_transformer_train.py --model vanilla --d_model 256 --nhead 8 --num_layers 4

# Run only folds 1-3 for a quick smoke-test
python loso_transformer_train.py --model patch --max_folds 3
"""

import os
import sys
import math
import pickle
import argparse
import warnings
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformer_models import build_model, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Constants (mirror MATLAB script)
# ---------------------------------------------------------------------------
FEATURE_ROWS   = list(range(1, 66)) + list(range(68, 83))   # 0-indexed (MATLAB 2:67,69:84 -> 1-indexed → subtract 1)
NUM_CLASSES    = 20
DOWNSAMPLE     = 2          # MATLAB: seq = A(:, 1:2:end) – take every 2nd column
AUG_SIGMA      = 0.01       # Gaussian noise std for training augmentation
VAL_FRAC       = 0.05       # stratified validation fraction
EXPECTED_ROWS  = 130        # skip files that don't have exactly 130 feature rows


CLASS_NAMES = [
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Hand Waving", "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting", "Phone Call", "Raising Hand", "Reading",
    "Scratching Arm", "Sitting Still", "Sit-to-Stand", "Standing",
    "Stand-to-Sit", "Stretching", "Thumb Sucking", "Walking",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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
    """Pad sequences to max length in the batch; return mask."""
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    C = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, C, dtype=torch.float32)
    mask   = torch.ones(len(seqs), max_len, dtype=torch.bool)  # True = padding
    for i, (s, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L] = s
        mask[i, :L]   = False   # False = real data

    return padded, torch.tensor(labels, dtype=torch.long), mask


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def extract_sequence(sample: dict, mu=None, sigma=None, augment: bool = False) -> torch.Tensor | None:
    """
    Extract feature-selected, downsampled sequence from a raw sample dict.
    Returns a (T, C) float32 tensor or None if the sample is invalid.
    """
    data = sample["data"]   # (total_features, time)
    if data.shape[0] != EXPECTED_ROWS:
        return None

    selected = data[FEATURE_ROWS, :]          # (C, T)
    if np.isnan(selected).any():
        return None

    seq = selected[:, ::DOWNSAMPLE].T.astype(np.float32)  # (T', C)

    if mu is not None:
        seq = (seq - mu) / sigma
    if augment:
        seq += np.random.randn(*seq.shape).astype(np.float32) * AUG_SIGMA

    return torch.from_numpy(seq)


def compute_normalization(sequences: list[torch.Tensor]):
    """Compute per-feature mean and std from a list of (T, C) tensors."""
    all_data = np.concatenate([s.numpy() for s in sequences], axis=0)  # (sum_T, C)
    mu    = all_data.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = all_data.std(axis=0, keepdims=True).astype(np.float32)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def apply_norm(seq_tensor: torch.Tensor, mu, sigma) -> torch.Tensor:
    return (seq_tensor.numpy() - mu) / sigma


def stratified_val_split(labels: list[int], val_frac: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(labels))
    labels_arr = np.array(labels)
    val_idx = []

    for c in range(NUM_CLASSES):
        idx_c = all_idx[labels_arr == c]
        n = len(idx_c)
        if n == 0:
            continue
        k = max(1, round(val_frac * n)) if n > 1 else 0
        if k > 0:
            chosen = rng.choice(idx_c, k, replace=False)
            val_idx.extend(chosen.tolist())

    val_idx = list(set(val_idx))
    if not val_idx:
        val_idx = rng.choice(all_idx, max(1, round(val_frac * len(labels))), replace=False).tolist()

    train_idx = [i for i in all_idx if i not in set(val_idx)]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, labels, mask in loader:
        seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(seqs, padding_mask=mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(seqs, padding_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for seqs, labels, mask in loader:
        seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
        logits = model(seqs, padding_mask=mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def compute_metrics(conf_mat: np.ndarray):
    """Compute per-class and macro metrics from confusion matrix."""
    TP = np.diag(conf_mat).astype(float)
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP
    TN = conf_mat.sum() - (TP + FP + FN)
    eps = 1e-10

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    mcc       = (TP * TN - FP * FN) / np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
    )
    acc = TP.sum() / conf_mat.sum()

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mcc":       mcc,
        "macro_precision": precision.mean(),
        "macro_recall":    recall.mean(),
        "macro_f1":        f1.mean(),
        "macro_mcc":       mcc.mean(),
    }


def save_confusion_matrix(conf_mat: np.ndarray, class_names: list, title: str, path: str):
    fig, ax = plt.subplots(figsize=(14, 12))
    # Row-normalize
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_cm  = np.divide(conf_mat, row_sums, where=row_sums != 0)
    sns.heatmap(
        norm_cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LOSO Main loop
# ---------------------------------------------------------------------------
def get_device():
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_adaptive_batch(
    model, train_dl_fn, val_dl_fn, optimizer_fn, scheduler_fn,
    criterion, args, device_str: str, start_batch: int
):
    """
    Mirror the MATLAB adaptive-batch fallback:
      1. Try training on GPU with the requested batch size.
      2. On CUDA OOM, halve the batch size and retry (down to min_batch=8).
      3. If all GPU attempts fail, fall back to CPU.

    Parameters
    ----------
    model        : nn.Module (already on device)
    train_dl_fn  : callable(batch_size) -> DataLoader  (recreates loader)
    val_dl_fn    : callable(batch_size) -> DataLoader
    optimizer_fn : callable(model) -> optimizer
    scheduler_fn : callable(optimizer) -> scheduler
    criterion    : loss function
    args         : Namespace (epochs, patience, log_interval)
    device_str   : "cuda" or "cpu"
    start_batch  : initial MiniBatchSize to try

    Returns
    -------
    model        : trained model (best checkpoint loaded)
    used_env     : "gpu" or "cpu"
    used_batch   : batch size that succeeded
    """
    MIN_BATCH = 8
    batch = start_batch

    while True:
        try:
            current_device = torch.device(device_str)
            model = model.to(current_device)
            optimizer = optimizer_fn(model)
            scheduler = scheduler_fn(optimizer)
            train_dl  = train_dl_fn(batch)
            val_dl    = val_dl_fn(batch)
            scaler    = torch.cuda.amp.GradScaler() if device_str == "cuda" else None

            best_val_acc = 0.0
            best_state   = None
            patience_cnt = 0

            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc = train_one_epoch(
                    model, train_dl, optimizer, criterion, current_device, scaler
                )
                vl_loss, vl_acc, _, _ = evaluate(model, val_dl, criterion, current_device)
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
            return model, current_device, used_env, batch

        except RuntimeError as e:
            msg = str(e).lower()
            is_oom = "out of memory" in msg or "cuda" in msg and "memory" in msg

            if is_oom and device_str == "cuda" and batch > MIN_BATCH:
                new_batch = max(MIN_BATCH, batch // 2)
                print(f"  [OOM] GPU out of memory at batch={batch}. "
                      f"Retrying with batch={new_batch}.")
                # Free GPU memory before retrying
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch = new_batch
                continue

            if is_oom and device_str == "cuda":
                print(f"  [OOM] GPU failed even at batch={batch}. Falling back to CPU.")
                device_str = "cpu"
                batch = start_batch   # reset batch size for CPU
                continue

            # Non-OOM error: re-raise
            raise


def run_loso(subjects: list, model_name: str, model_kwargs: dict, args):
    device_str   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device_str.upper()}")

    num_subjects = len(subjects)
    max_folds    = args.max_folds if args.max_folds > 0 else num_subjects

    # Output directories
    run_tag  = model_name
    cm_dir   = Path(args.out_dir) / run_tag / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    excel_cm = Path(args.out_dir) / run_tag / "LOSO_ConfusionMatrices.xlsx"
    excel_mt = Path(args.out_dir) / run_tag / "LOSO_PerClassMetrics.xlsx"

    # Issue 4 fix: delete stale Excel files at the start of each run so
    # rerunning never hits a duplicate-sheet error.
    for xf in (excel_cm, excel_mt):
        if xf.exists():
            xf.unlink()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    fold_metrics = []

    for s_idx in range(min(num_subjects, max_folds)):
        subj = subjects[s_idx]
        sid  = subj["subjectID"]
        print(f"\n=== Leave subject {s_idx+1}/{num_subjects} out: {sid} ===")

        # ---- Build raw sequences ----
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

        # ---- Stratified val split ----
        train_idx, val_idx = stratified_val_split(pool_labels, VAL_FRAC, seed=s_idx + 1)
        train_seqs_raw = [pool_seqs[i] for i in train_idx]
        train_labels   = [pool_labels[i] for i in train_idx]
        val_seqs_raw   = [pool_seqs[i] for i in val_idx]
        val_labels     = [pool_labels[i] for i in val_idx]

        # ---- Normalize ----
        mu, sigma = compute_normalization(train_seqs_raw + val_seqs_raw)

        def norm_aug(seqs, augment=False):
            out = []
            for s in seqs:
                n = (s.numpy() - mu) / sigma
                if augment:
                    n = n + np.random.randn(*n.shape).astype(np.float32) * AUG_SIGMA
                out.append(torch.from_numpy(n))
            return out

        train_seqs  = norm_aug(train_seqs_raw, augment=True)
        val_seqs    = norm_aug(val_seqs_raw,   augment=False)
        test_seqs_n = norm_aug(test_seqs,      augment=False)

        # ---- Infer seq_len for models that need it ----
        all_lens = [s.shape[0] for s in train_seqs + val_seqs + test_seqs_n]
        max_len  = max(all_lens)

        # ---- DataLoader factories (needed for adaptive-batch retry) ----
        pin = device_str == "cuda"
        train_ds = SequenceDataset(train_seqs, train_labels)
        val_ds   = SequenceDataset(val_seqs,   val_labels)
        test_ds  = SequenceDataset(test_seqs_n, test_labels)

        def make_train_dl(bs):
            return DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=pin)
        def make_val_dl(bs):
            return DataLoader(val_ds, batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

        # ---- Build model ----
        input_dim = train_seqs[0].shape[1]
        model = build_model(model_name, input_dim, NUM_CLASSES, max_len, **model_kwargs)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {model_name}  |  Params: {n_params:,}  |  Input dim: {input_dim}")

        # ---- Issue 2: adaptive GPU->CPU batch fallback ----
        def make_optimizer(m):
            return torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        def make_scheduler(opt):
            return CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

        model, device, used_env, used_batch = train_with_adaptive_batch(
            model, make_train_dl, make_val_dl,
            make_optimizer, make_scheduler,
            criterion, args, device_str, args.batch_size
        )
        print(f"  Trained on {used_env.upper()} with batch_size={used_batch}")

        # ---- Test ----
        model.to(device)
        _, test_acc, y_pred, y_true = evaluate(model, test_dl, criterion, device)
        print(f"  Subject {s_idx+1} test accuracy: {test_acc*100:.2f}%")

        # ---- Confusion matrix & metrics ----
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

        # ---- Save confusion matrix PNG ----
        safe_id = "".join(c if c.isalnum() else "_" for c in sid)
        cm_path = str(cm_dir / f"CM_Subject_{s_idx+1:02d}_{safe_id}.png")
        save_confusion_matrix(
            conf_mat, CLASS_NAMES,
            title=f"Confusion Matrix – Subject {s_idx+1} ({sid})  [{model_name}]",
            path=cm_path
        )

        # ---- Issue 4 fix: write Excel with if_sheet_exists="replace" ----
        sheet = f"Subject_{s_idx+1}"
        cm_df = pd.DataFrame(conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        mode = "a" if excel_cm.exists() else "w"
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

    # ---- Summary ----
    if fold_metrics:
        print(f"\n=== Summary [{model_name}] over {len(fold_metrics)} folds ===")
        for k in ["accuracy", "precision", "recall", "f1", "mcc"]:
            avg = np.mean([m[k] for m in fold_metrics])
            print(f"  Average {k:10s}: {avg:.4f}")

        summary_df = pd.DataFrame(fold_metrics)
        summary_path = Path(args.out_dir) / run_tag / "LOSO_Summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved -> {summary_path}")

    return fold_metrics



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO Transformer Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data & output ─────────────────────────────────────────────────────
    p.add_argument("--data",         default="subjects_per_class.pkl",
                   help="Pickle file from load_cell_data.py")
    p.add_argument("--model",        default="conformer",
                   choices=list(MODEL_REGISTRY) + ["all"],
                   help="Transformer variant to use, or 'all' to sweep")
    p.add_argument("--out_dir",      default="loso_results")
    # ── Training knobs ────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32,
                   help="Starting batch size; halved on GPU OOM before falling back to CPU")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=20,
                   help="Early-stopping patience (epochs without val-acc improvement)")
    p.add_argument("--log_interval", type=int,   default=10)
    p.add_argument("--max_folds",    type=int,   default=0,
                   help="Limit LOSO folds (0 = all subjects)")
    # ── Shared model hyper-params ─────────────────────────────────────────
    p.add_argument("--d_model",         type=int,   default=None,
                   help="Embedding dimension [all models]")
    p.add_argument("--nhead",           type=int,   default=None,
                   help="Self-attention heads [all models]")
    p.add_argument("--num_layers",      type=int,   default=None,
                   help="Encoder depth [all models]")
    p.add_argument("--dim_feedforward", type=int,   default=None,
                   help="FFN hidden size [vanilla, patch, conv_stem]")
    p.add_argument("--dropout",         type=float, default=None,
                   help="Dropout rate [all models]")
    # ── Variant-specific hyper-params ─────────────────────────────────────
    p.add_argument("--patch_size",   type=int,   default=None,
                   help="Time steps per patch [patch]")
    p.add_argument("--kernel_size",  type=int,   default=None,
                   help="Conv kernel size [conv_stem, conformer]")
    p.add_argument("--k",            type=int,   default=None,
                   help="Low-rank projection dim [linformer]")
    p.add_argument("--window_size",  type=int,   default=None,
                   help="Local window length [hierarchical]")
    p.add_argument("--d_local",      type=int,   default=None,
                   help="Local-stage embedding dim [hierarchical]")
    p.add_argument("--d_global",     type=int,   default=None,
                   help="Global-stage embedding dim [hierarchical]")
    p.add_argument("--num_latents",  type=int,   default=None,
                   help="Number of learnable query vectors [cross_attention]")
    p.add_argument("--d_latent",     type=int,   default=None,
                   help="Latent query dimension [cross_attention]")
    p.add_argument("--nhead_cross",  type=int,   default=None,
                   help="Cross-attention heads [cross_attention]")
    return p.parse_args()


def collect_model_kwargs(args) -> dict:
    all_keys = [
        # shared
        "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
        # variant-specific
        "patch_size", "kernel_size", "k",
        "window_size", "d_local", "d_global",
        "num_latents", "d_latent", "nhead_cross",
    ]
    return {k: getattr(args, k) for k in all_keys if getattr(args, k, None) is not None}


def main():
    args = parse_args()

    # Load data
    if not Path(args.data).exists():
        print(f"[ERROR] Data file not found: {args.data}")
        print("  Run load_cell_data.py first.")
        sys.exit(1)

    with open(args.data, "rb") as f:
        subjects = pickle.load(f)
    print(f"Loaded {len(subjects)} subjects from {args.data}")

    model_kwargs = collect_model_kwargs(args)
    models_to_run = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    all_results = {}
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        results = run_loso(subjects, model_name, model_kwargs, args)
        all_results[model_name] = results

    # Cross-model comparison summary
    if len(models_to_run) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-MODEL COMPARISON")
        print(f"{'='*60}")
        rows = []
        for mname, folds in all_results.items():
            if not folds:
                continue
            rows.append({
                "model":     mname,
                "accuracy":  np.mean([m["accuracy"]  for m in folds]),
                "precision": np.mean([m["precision"] for m in folds]),
                "recall":    np.mean([m["recall"]    for m in folds]),
                "f1":        np.mean([m["f1"]        for m in folds]),
                "mcc":       np.mean([m["mcc"]       for m in folds]),
            })
        comp_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
        print(comp_df.to_string(index=False))
        comp_path = Path(args.out_dir) / "model_comparison.csv"
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(comp_path, index=False)
        print(f"\nComparison table saved -> {comp_path}")


if __name__ == "__main__":
    main()
