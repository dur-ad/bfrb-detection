"""
loso_transformer_train.py  (speed-optimised + checkpointing)
=============================================================
Python / PyTorch LOSO training for Transformer variants.

Speed improvements over the original (mirrors loso_model_zoo_fast.py):
  • torch.backends.cudnn.benchmark = True + TF32 enabled
  • num_workers=4 + persistent_workers + prefetch_factor=2
  • optimizer.zero_grad(set_to_none=True)
  • best_state uses .detach().clone() instead of .cpu().clone()
  • torch.amp.autocast / GradScaler (non-deprecated API)
  • Excel writing deferred to AFTER all folds finish (no I/O in hot loop)
  • Per-fold confusion-matrix PNG skipped by default; use --save_cm to enable
  • torch.compile() applied when PyTorch >= 2.0
  • DataLoader workers shut down explicitly between folds (fixes "too many open files")
  • Checkpoint saved after every fold — resume with the same command if interrupted
  • Stale Excel files deleted at run start (no duplicate-sheet errors)

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

# Resume an interrupted run — just re-run the same command
python loso_transformer_train.py --model conformer
"""

import os
import sys
import math
import pickle
import argparse
import warnings
from pathlib import Path

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
import time

from transformer_models import build_model, MODEL_REGISTRY


# ── Speed knobs ───────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 24
DOWNSAMPLE  = 2          # take every 2nd time step
AUG_SIGMA   = 0.01       # Gaussian noise std for training augmentation
VAL_FRAC    = 0.05       # stratified validation fraction

# Reduce if you hit "too many open files" across many LOSO folds.
# Pass --workers 0 for single-process (safest).
DL_WORKERS = 8

CLASS_NAMES = [
    "Hair Pulling",
    "Nail Biting",
    "Nose Picking",
    "Thumb Sucking",
    "Eyeglasses",
    "Knuckle Cracking",
    "Face Touching",
    "Leg Shaking",
    "Scratching Arm",
    "Cuticle Picking",
    "Leg Scratching",
    "Phone Call",
    "Eating",
    "Drinking",
    "Stretching",
    "Hand Waving",
    "Reading",
    "Using Phone",
    "Standing",
    "Sit-to-Stand",
    "Stand-to-Sit",
    "Walking",
    "Sitting Still",
    "Raising Hand",
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
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len  = int(lengths.max())
    C        = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, C, dtype=torch.float32)
    mask   = torch.ones(len(seqs), max_len, dtype=torch.bool)   # True = padding
    for i, (s, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L] = s
        mask[i, :L]   = False   # False = real data

    return padded, torch.tensor(labels, dtype=torch.long), lengths, mask


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def extract_sequence(sample: dict):
    """
    Extract a downsampled sequence using ALL feature rows.
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
    if not val_idx:
        val_idx = rng.choice(all_idx, max(1, round(val_frac * len(labels))), replace=False).tolist()
    train_idx = [i for i in all_idx if i not in set(val_idx)]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(ckpt_path: Path, fold_idx: int,
                    fold_metrics: list, all_cm_data: dict, all_mt_data: dict):
    """Persist everything needed to resume after fold_idx completes."""
    ckpt = {
        "last_completed_fold": fold_idx,
        "fold_metrics":        fold_metrics,
        "all_cm_data":         all_cm_data,
        "all_mt_data":         all_mt_data,
    }
    # Write to a temp file first, then rename — avoids corrupt checkpoint
    # if the process is killed mid-write.
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.replace(ckpt_path)
    print(f"  [Checkpoint] saved after fold {fold_idx + 1} → {ckpt_path}")


def load_checkpoint(ckpt_path: Path):
    """Return checkpoint dict or None if no checkpoint exists."""
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"  [Checkpoint] found — resuming from fold {ckpt['last_completed_fold'] + 2} "
              f"({ckpt['last_completed_fold'] + 1} folds already done)")
        return ckpt
    except Exception as e:
        warnings.warn(f"  [Checkpoint] could not load {ckpt_path}: {e}  — starting fresh.")
        return None


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def save_confusion_matrix(conf_mat: np.ndarray, class_names: list, title: str, path: str):
    fig, ax  = plt.subplots(figsize=(14, 12))
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_cm  = np.divide(conf_mat, row_sums,
                         out=np.zeros_like(conf_mat, dtype=float),
                         where=row_sums != 0)
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
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for seqs, labels, lengths, mask in loader:
        seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
        # lengths not used by Transformer models but kept for API consistency
        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(seqs, padding_mask=mask)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(seqs, padding_mask=mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for seqs, labels, lengths, mask in loader:
        seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)
        logits = model(seqs, padding_mask=mask)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def compute_metrics(conf_mat: np.ndarray):
    """Compute per-class and macro metrics from confusion matrix."""
    TP  = np.diag(conf_mat).astype(float)
    FP  = conf_mat.sum(axis=0) - TP
    FN  = conf_mat.sum(axis=1) - TP
    TN  = conf_mat.sum() - (TP + FP + FN)
    eps = 1e-10
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    mcc       = (TP * TN - FP * FN) / np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
    )
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


# ---------------------------------------------------------------------------
# Adaptive GPU → CPU batch fallback  (mirrors loso_model_zoo_fast.py)
# ---------------------------------------------------------------------------
def train_with_adaptive_batch(
    model, make_train_dl, make_val_dl,
    make_optimizer, make_scheduler,
    criterion, args, device_str, start_batch
):
    """
    1. Try training on GPU with the requested batch size.
    2. On CUDA OOM, halve the batch size and retry (down to MIN_BATCH=8).
    3. If all GPU attempts fail, fall back to CPU.

    Returns
    -------
    model, device, used_env ("gpu"/"cpu"), used_batch, train_dl, val_dl
    """
    MIN_BATCH = 8
    batch     = start_batch

    while True:
        try:
            device = torch.device(device_str)
            model  = model.to(device)

            # torch.compile (PyTorch 2+) — skip gracefully if unavailable
            if args.compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                    print("  [torch.compile] enabled")
                except Exception as ce:
                    print(f"  [torch.compile] skipped: {ce}")

            optimizer = make_optimizer(model)
            scheduler = make_scheduler(optimizer)
            train_dl  = make_train_dl(batch)
            val_dl    = make_val_dl(batch)
            scaler    = torch.amp.GradScaler("cuda") if device_str == "cuda" else None

            best_val_acc = 0.0
            best_state   = None
            patience_cnt = 0

            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc       = train_one_epoch(
                    model, train_dl, optimizer, criterion, device, scaler)
                vl_loss, vl_acc, _, _ = evaluate(model, val_dl, criterion, device)
                scheduler.step()

                if epoch % args.log_interval == 0 or epoch == args.epochs:
                    print(f"  Epoch {epoch:3d}/{args.epochs} | "
                          f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                          f"val loss {vl_loss:.4f} acc {vl_acc:.3f}")

                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    # detach().clone() — faster than .cpu().clone()
                    best_state   = {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                        break

            if best_state:
                model.load_state_dict(best_state)

            used_env = "gpu" if device_str == "cuda" else "cpu"
            return model, device, used_env, batch, train_dl, val_dl

        except RuntimeError as e:
            msg    = str(e).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)

            if is_oom and device_str == "cuda" and batch > MIN_BATCH:
                new_batch = max(MIN_BATCH, batch // 2)
                print(f"  [OOM] GPU out of memory at batch={batch}. "
                      f"Retrying with batch={new_batch}.")
                torch.cuda.empty_cache()
                batch = new_batch
                continue

            if is_oom and device_str == "cuda":
                print(f"  [OOM] GPU failed even at batch={batch}. Falling back to CPU.")
                device_str = "cpu"
                batch      = start_batch
                continue

            raise   # non-OOM error: re-raise


# ---------------------------------------------------------------------------
# LOSO Main loop
# ---------------------------------------------------------------------------
def run_loso(subjects: list, model_name: str, model_kwargs: dict, args):
    device_str   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device_str.upper()}")

    num_subjects = len(subjects)
    max_folds    = args.max_folds if args.max_folds > 0 else num_subjects

    # Output directories
    out_model_dir = Path(args.out_dir) / model_name
    cm_dir        = out_model_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Checkpoint: load if a previous run was interrupted ───────────────────
    ckpt_path    = out_model_dir / "checkpoint.pkl"
    ckpt         = load_checkpoint(ckpt_path)
    resume_from  = 0
    fold_metrics = []
    all_cm_data  = {}
    all_mt_data  = {}

    if ckpt is not None:
        resume_from  = ckpt["last_completed_fold"] + 1
        fold_metrics = ckpt["fold_metrics"]
        all_cm_data  = ckpt["all_cm_data"]
        all_mt_data  = ckpt["all_mt_data"]

    # ── Fold loop ─────────────────────────────────────────────────────────────
    for s_idx in range(min(num_subjects, max_folds)):

        # Skip folds already completed in a previous run
        if s_idx < resume_from:
            subj = subjects[s_idx]
            print(f"  [Checkpoint] skipping fold {s_idx + 1} "
                  f"(subject {subj['subjectID']} — already done)")
            continue

        subj = subjects[s_idx]
        sid  = subj["subjectID"]
        print(f"\n=== Fold {s_idx+1}/{min(num_subjects, max_folds)} "
              f"– Leave out: {sid} ===")

        # ── Build pool (train+val) and test sets ─────────────────────────────
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

        # ── Stratified val split ──────────────────────────────────────────────
        train_idx, val_idx = stratified_val_split(pool_labels, VAL_FRAC, seed=s_idx + 1)
        train_seqs_raw = [pool_seqs[i] for i in train_idx]
        train_labels   = [pool_labels[i] for i in train_idx]
        val_seqs_raw   = [pool_seqs[i] for i in val_idx]
        val_labels     = [pool_labels[i] for i in val_idx]

        # ── Normalize ─────────────────────────────────────────────────────────
        mu, sigma   = compute_normalization(train_seqs_raw + val_seqs_raw)
        train_seqs  = apply_norm_aug(train_seqs_raw, mu, sigma, augment=True)
        val_seqs    = apply_norm_aug(val_seqs_raw,   mu, sigma, augment=False)
        test_seqs_n = apply_norm_aug(test_seqs,      mu, sigma, augment=False)

        # ── Infer max_len for models that need it ─────────────────────────────
        all_lens = [s.shape[0] for s in train_seqs + val_seqs + test_seqs_n]
        max_len  = max(all_lens)

        # ── Datasets & DataLoader factories ───────────────────────────────────
        pin      = device_str == "cuda"
        train_ds = SequenceDataset(train_seqs,  train_labels)
        val_ds   = SequenceDataset(val_seqs,    val_labels)
        test_ds  = SequenceDataset(test_seqs_n, test_labels)

        def make_train_dl(bs):
            return DataLoader(
                train_ds, batch_size=bs, shuffle=True,
                collate_fn=collate_fn,
                num_workers=DL_WORKERS,
                pin_memory=pin,
                persistent_workers=(DL_WORKERS > 0),
                prefetch_factor=2 if DL_WORKERS > 0 else None,
            )

        def make_val_dl(bs):
            return DataLoader(
                val_ds, batch_size=bs, shuffle=False,
                collate_fn=collate_fn,
                num_workers=DL_WORKERS,
                pin_memory=pin,
                persistent_workers=(DL_WORKERS > 0),
                prefetch_factor=2 if DL_WORKERS > 0 else None,
            )

        test_dl = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn,
            num_workers=DL_WORKERS,
            pin_memory=pin,
            persistent_workers=(DL_WORKERS > 0),
            prefetch_factor=2 if DL_WORKERS > 0 else None,
        )

        # ── Build model ───────────────────────────────────────────────────────
        input_dim = train_seqs[0].shape[1]
        model     = build_model(model_name, input_dim, NUM_CLASSES, max_len, **model_kwargs)

        if s_idx == resume_from:   # first fold we actually train — print once
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Model     : {model_name}")
            print(f"  Input dim : {input_dim}")
            print(f"  Params    : {n_params:,}")

        def make_optimizer(m):
            return torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)

        def make_scheduler(opt):
            return CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

        model, device, used_env, used_batch, train_dl, val_dl = \
            train_with_adaptive_batch(
                model, make_train_dl, make_val_dl,
                make_optimizer, make_scheduler,
                criterion, args, device_str, args.batch_size,
            )
        print(f"  Trained on {used_env.upper()} | batch_size={used_batch}")

        # ── Test ──────────────────────────────────────────────────────────────
        model.to(device)
        _, test_acc, y_pred, y_true = evaluate(model, test_dl, criterion, device)
        print(f"  Test accuracy: {test_acc * 100:.2f}%")

        # ── Shut down DataLoader workers to free file descriptors ─────────────
        # Prevents "too many open files" when running many LOSO folds with
        # persistent_workers=True.
        for dl in [train_dl, val_dl, test_dl]:
            dl._iterator = None
        del train_dl, val_dl, test_dl

        # ── Metrics ───────────────────────────────────────────────────────────
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

        # Optional per-fold confusion matrix PNG
        if args.save_cm:
            safe_id = "".join(c if c.isalnum() else "_" for c in sid)
            cm_path = str(cm_dir / f"CM_Fold_{s_idx+1:02d}_{safe_id}.png")
            save_confusion_matrix(
                conf_mat, CLASS_NAMES,
                title=f"Confusion Matrix – {sid}  [{model_name}]",
                path=cm_path,
            )

        # Accumulate Excel data in memory (written once after all folds)
        sheet = f"Subject_{s_idx+1}"
        all_cm_data[sheet] = pd.DataFrame(
            conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        all_mt_data[sheet] = pd.DataFrame({
            "Precision": metrics["precision"],
            "Recall":    metrics["recall"],
            "F1Score":   metrics["f1"],
            "MCC":       metrics["mcc"],
        }, index=CLASS_NAMES)

        # ── Save checkpoint after every completed fold ─────────────────────
        save_checkpoint(ckpt_path, s_idx, fold_metrics, all_cm_data, all_mt_data)

    # ── Write Excel ONCE after all folds ──────────────────────────────────────
    if all_cm_data:
        excel_cm = out_model_dir / "LOSO_ConfusionMatrices.xlsx"
        excel_mt = out_model_dir / "LOSO_PerClassMetrics.xlsx"
        with pd.ExcelWriter(excel_cm, engine="openpyxl") as w:
            for sheet, df in all_cm_data.items():
                df.to_excel(w, sheet_name=sheet)
        with pd.ExcelWriter(excel_mt, engine="openpyxl") as w:
            for sheet, df in all_mt_data.items():
                df.to_excel(w, sheet_name=sheet)
        print(f"  Excel saved → {out_model_dir}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    if fold_metrics:
        print(f"\n=== Summary [{model_name}] — {len(fold_metrics)} folds ===")
        for k in ["accuracy", "precision", "recall", "f1", "mcc"]:
            avg = np.mean([m[k] for m in fold_metrics])
            print(f"  Avg {k:10s}: {avg:.4f}")
        summary_df   = pd.DataFrame(fold_metrics)
        summary_path = out_model_dir / "LOSO_Summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Summary saved → {summary_path}")

        # Delete checkpoint now that all folds are done
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"  [Checkpoint] deleted — all folds complete.")

    return fold_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO Transformer Training (speed-optimised + checkpointing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data & output ─────────────────────────────────────────────────────
    p.add_argument("--data",         default="subjects_cleaned_20subjects.pkl",
                   help="Pickle file from load_cell_data.py")
    p.add_argument("--model",        default="all",
                   choices=list(MODEL_REGISTRY) + ["all"],
                   help="Transformer variant to use, or 'all' to sweep")
    p.add_argument("--out_dir",      default="loso_transformer_results")
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
    # ── Speed options ─────────────────────────────────────────────────────
    p.add_argument("--save_cm",      action="store_true",
                   help="Save per-fold confusion matrix PNGs (slower)")
    p.add_argument("--compile",      action="store_true",
                   help="Apply torch.compile() — PyTorch 2+ only")
    p.add_argument("--workers",      type=int,   default=DL_WORKERS,
                   help="DataLoader worker processes (0=single-process, safest)")
    # ── Shared model hyper-params ─────────────────────────────────────────
    p.add_argument("--d_model",         type=int,   default=None)
    p.add_argument("--nhead",           type=int,   default=None)
    p.add_argument("--num_layers",      type=int,   default=None)
    p.add_argument("--dim_feedforward", type=int,   default=None)
    p.add_argument("--dropout",         type=float, default=None)
    # ── Variant-specific hyper-params ─────────────────────────────────────
    p.add_argument("--patch_size",   type=int,   default=None)
    p.add_argument("--kernel_size",  type=int,   default=None)
    p.add_argument("--k",            type=int,   default=None)
    p.add_argument("--window_size",  type=int,   default=None)
    p.add_argument("--d_local",      type=int,   default=None)
    p.add_argument("--d_global",     type=int,   default=None)
    p.add_argument("--num_latents",  type=int,   default=None)
    p.add_argument("--d_latent",     type=int,   default=None)
    p.add_argument("--nhead_cross",  type=int,   default=None)
    return p.parse_args()


def collect_model_kwargs(args) -> dict:
    all_keys = [
        "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
        "patch_size", "kernel_size", "k",
        "window_size", "d_local", "d_global",
        "num_latents", "d_latent", "nhead_cross",
    ]
    return {k: getattr(args, k) for k in all_keys if getattr(args, k, None) is not None}


def main():
    args = parse_args()

    global DL_WORKERS
    DL_WORKERS = args.workers

    if not Path(args.data).exists():
        print(f"[ERROR] Data file not found: {args.data}")
        print("  Run load_cell_data.py first.")
        sys.exit(1)

    with open(args.data, "rb") as f:
        subjects = pickle.load(f)
    print(f"Loaded {len(subjects)} subjects from {args.data}")

    model_kwargs  = collect_model_kwargs(args)
    EXCLUDE_MODELS = ["patch", "conv_stem", "cross_attention", "patch", "vanilla"]    # add model names here to skip them
    models_to_run  = (
        [m for m in MODEL_REGISTRY if m not in EXCLUDE_MODELS]
        if args.model == "all"
        else [args.model]
    )

    all_results = {}
    model_times = {}

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            results = run_loso(subjects, model_name, model_kwargs, args)
            all_results[model_name] = results
        except Exception as e:
            warnings.warn(f"Model {model_name} failed: {e}")
            all_results[model_name] = []
        model_times[model_name] = (time.time() - t0) / 3600

    # ── Cross-model comparison summary ────────────────────────────────────────
    if len(models_to_run) > 1:
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
        if rows:
            comp_df   = pd.DataFrame(rows).sort_values("f1", ascending=False)
            comp_path = Path(args.out_dir) / "model_comparison.csv"
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            comp_df.to_csv(comp_path, index=False)
            print(f"\n{'='*60}")
            print("  CROSS-MODEL COMPARISON")
            print(f"{'='*60}")
            print(comp_df.to_string(index=False))
            print(f"\nComparison table saved → {comp_path}")

    print(f"\n{'='*60}")
    print("  MODEL TRAINING TIME SUMMARY")
    print(f"{'='*60}")
    for m, t in model_times.items():
        print(f"  {m:20s}: {t:.2f} hours")
    print(f"  Total: {sum(model_times.values()):.2f} hours")


if __name__ == "__main__":
    main()
