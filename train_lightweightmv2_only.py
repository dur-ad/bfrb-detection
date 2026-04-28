"""
train_lightweightmv2.py
=======================
1. Runs LOSO cross-validation on subjects_cleaned_20subjects.pkl
   → produces per-fold metrics, confusion matrices, summary CSV

2. Retrains LightweightMV2 on ALL subjects / ALL samples using the
   same hyperparameters (no held-out data) → this is the exported model

3. Exports the final model to model_export/:
    model_export/
    ├── model.onnx
    ├── model_state_dict.pt
    ├── model_class.py
    ├── load_and_test.py
    └── README.md

GPU Optimisations (mirrored from loso_model_zoo_fast.py)
---------------------------------------------------------
  • torch.backends.cudnn.benchmark = True
  • TF32 enabled on Ampere+ GPUs
  • num_workers=12 + persistent_workers + prefetch_factor=2
  • optimizer.zero_grad(set_to_none=True)
  • best_state uses .detach().clone()
  • torch.amp.autocast / GradScaler (updated non-deprecated API)
  • Excel writing deferred until after all folds
  • Checkpoint saved after every LOSO fold (resume-safe)
  • DataLoader workers shut down between folds (no "too many open files")
  • torch.compile() support (PyTorch 2+)

Usage
-----
    python train_lightweightmv2.py --skip_loso
    python train_lightweightmv2.py --epochs 100 --batch_size 64
    python train_lightweightmv2.py --skip_loso --epochs 150 --batch_size 64
"""

import argparse
import os
import pickle
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
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Speed knobs ───────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 24
DOWNSAMPLE  = 2
AUG_SIGMA   = 0.01
VAL_FRAC    = 0.05
MODEL_NAME  = "LightweightMV2"
OUTPUT_DIR  = "LightweightMV2_loso_results_20subjects"
EXPORT_DIR  = "model_export"
DL_WORKERS  = 12

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

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class _InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid          = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1),
            nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, mid, 3, stride, 1, groups=mid),
            nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, out_ch, 1),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        return (x + self.conv(x)) if self.use_res else self.conv(x)


class LightweightMV2(nn.Module):
    """
    MobileNet-V2-style depthwise separable 1-D CNN.
    Input  : (B, T, C)  — batch x time x features
    Output : (B, num_classes) logits
    """
    def __init__(self, input_size, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        cfg = [
            (32,  1, 1, 1),
            (48,  2, 6, 2),
            (64,  1, 6, 3),
            (96,  2, 6, 2),
            (128, 1, 6, 1),
        ]
        layers = [nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU6(True),
        )]
        in_ch = 32
        for out_ch, s, e, n in cfg:
            for i in range(n):
                layers.append(_InvertedResidual1D(
                    in_ch, out_ch, stride=s if i == 0 else 1, expand=e))
                in_ch = out_ch
        layers += [nn.Conv1d(in_ch, 256, 1), nn.BatchNorm1d(256), nn.ReLU6(True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths=None, padding_mask=None):
        return self.head(self.features(x.permute(0, 2, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET & DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels    = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len  = int(lengths.max())
    C        = seqs[0].shape[1]
    padded   = torch.zeros(len(seqs), max_len, C, dtype=torch.float32)
    pad_mask = torch.ones(len(seqs), max_len, dtype=torch.bool)
    for i, (s, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L]   = s
        pad_mask[i, :L] = False
    return padded, torch.tensor(labels, dtype=torch.long), lengths, pad_mask


def extract_sequence(sample):
    data = sample["data"]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        return None
    if np.isnan(data).any():
        return None
    return torch.from_numpy(data[:, ::DOWNSAMPLE].T.astype(np.float32))


def compute_normalization(sequences):
    all_data = np.concatenate([s.numpy() for s in sequences], axis=0)
    mu    = all_data.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = all_data.std(axis=0,  keepdims=True).astype(np.float32)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def apply_norm_aug(seqs, mu, sigma, augment=False):
    out = []
    for s in seqs:
        n = (s.numpy() - mu) / sigma
        if augment:
            n = n + np.random.randn(*n.shape).astype(np.float32) * AUG_SIGMA
        out.append(torch.from_numpy(n))
    return out


def stratified_val_split(labels, val_frac, seed=1):
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
# CHECKPOINT HELPERS  (new — matches loso_model_zoo_fast.py)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(ckpt_path: Path, fold_idx: int,
                    fold_metrics: list, all_cm_data: dict, all_mt_data: dict):
    """Atomically persist progress after each LOSO fold."""
    ckpt = {
        "last_completed_fold": fold_idx,
        "fold_metrics":        fold_metrics,
        "all_cm_data":         all_cm_data,
        "all_mt_data":         all_mt_data,
    }
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.replace(ckpt_path)
    print(f"  [Checkpoint] saved after fold {fold_idx + 1} → {ckpt_path}")


def load_checkpoint(ckpt_path: Path):
    """Return checkpoint dict or None if none exists."""
    if not ckpt_path.exists():
        return None
    try:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"  [Checkpoint] found — resuming from fold "
              f"{ckpt['last_completed_fold'] + 2} "
              f"({ckpt['last_completed_fold'] + 1} folds already done)")
        return ckpt
    except Exception as e:
        warnings.warn(f"  [Checkpoint] could not load {ckpt_path}: {e} — starting fresh.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y, lengths, pad_mask in loader:
        x, y     = x.to(device), y.to(device)
        lengths  = lengths.to(device)
        pad_mask = pad_mask.to(device)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        logits   = model(x)
        loss     = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def compute_metrics(conf_mat):
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
        "precision":       precision, "recall": recall,
        "f1":              f1,        "mcc":    mcc,
        "macro_precision": precision.mean(), "macro_recall": recall.mean(),
        "macro_f1":        f1.mean(),        "macro_mcc":   mcc.mean(),
    }


def save_confusion_matrix(conf_mat, class_names, title, path):
    fig, ax  = plt.subplots(figsize=(14, 12))
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_cm  = np.divide(conf_mat, row_sums,
                         out=np.zeros_like(conf_mat, dtype=float),
                         where=row_sums != 0)
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
# SHARED CORE TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _do_train(model, make_train_dl, make_val_dl, make_optimizer, make_scheduler,
              criterion, args, device_str, start_batch, use_early_stopping=True):
    """
    use_early_stopping=True  → LOSO mode: val split, save best checkpoint
    use_early_stopping=False → retrain mode: run exactly args.epochs
    """
    MIN_BATCH = 8
    batch     = start_batch

    while True:
        try:
            device    = torch.device(device_str)
            model     = model.to(device)

            # torch.compile() support (PyTorch 2+)
            if args.compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                    print("  [torch.compile] enabled")
                except Exception as ce:
                    print(f"  [torch.compile] skipped: {ce}")

            optimizer = make_optimizer(model)
            scheduler = make_scheduler(optimizer)
            train_dl  = make_train_dl(batch)
            val_dl    = make_val_dl(batch) if use_early_stopping else None
            scaler    = torch.amp.GradScaler("cuda") if device_str == "cuda" else None

            best_val_acc = 0.0
            best_state   = None
            patience_cnt = 0

            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc = train_one_epoch(
                    model, train_dl, optimizer, criterion, device, scaler)
                scheduler.step()

                if use_early_stopping and val_dl is not None:
                    vl_loss, vl_acc, _, _ = evaluate(model, val_dl, criterion, device)
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
                            print(f"  Early stop at epoch {epoch} "
                                  f"(patience={args.patience})")
                            break
                else:
                    if epoch % args.log_interval == 0 or epoch == args.epochs:
                        print(f"  Epoch {epoch:3d}/{args.epochs} | "
                              f"train loss {tr_loss:.4f}  acc {tr_acc:.3f}")

            if use_early_stopping and best_state is not None:
                model.load_state_dict(best_state)

            return model, device, "gpu" if device_str == "cuda" else "cpu", batch, train_dl, val_dl

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
            raise


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — LOSO CROSS-VALIDATION  (with checkpointing)
# ─────────────────────────────────────────────────────────────────────────────

def run_loso(subjects, args):
    device_str   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device_str.upper()}")
    num_subjects = len(subjects)
    max_folds    = args.max_folds if args.max_folds > 0 else num_subjects

    out_dir = Path(args.out_dir)
    cm_dir  = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Checkpoint: resume if interrupted ────────────────────────────────────
    ckpt_path    = out_dir / "checkpoint.pkl"
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

    pin = device_str == "cuda"

    for s_idx in range(min(num_subjects, max_folds)):

        if s_idx < resume_from:
            subj = subjects[s_idx]
            print(f"  [Checkpoint] skipping fold {s_idx + 1} "
                  f"(subject {subj['subjectID']} — already done)")
            continue

        subj = subjects[s_idx]
        sid  = subj["subjectID"]
        print(f"\n{'='*60}")
        print(f"  Fold {s_idx+1}/{min(num_subjects, max_folds)} — Leave out: {sid}")
        print(f"{'='*60}")

        pool_seqs, pool_labels = [], []
        for i, other in enumerate(subjects):
            if i == s_idx:
                continue
            for smp in other["samples"]:
                seq = extract_sequence(smp)
                if seq is not None:
                    pool_seqs.append(seq)
                    pool_labels.append(smp["label"] - 1)

        test_seqs, test_labels = [], []
        for smp in subj["samples"]:
            seq = extract_sequence(smp)
            if seq is not None:
                test_seqs.append(seq)
                test_labels.append(smp["label"] - 1)

        if not pool_seqs or not test_seqs:
            warnings.warn(f"Fold {s_idx+1} empty — skipping.")
            continue

        train_idx, val_idx = stratified_val_split(pool_labels, VAL_FRAC, seed=s_idx+1)
        train_seqs_raw = [pool_seqs[i] for i in train_idx]
        train_labels   = [pool_labels[i] for i in train_idx]
        val_seqs_raw   = [pool_seqs[i] for i in val_idx]
        val_labels     = [pool_labels[i] for i in val_idx]

        mu, sigma   = compute_normalization(train_seqs_raw + val_seqs_raw)
        train_seqs  = apply_norm_aug(train_seqs_raw, mu, sigma, augment=True)
        val_seqs    = apply_norm_aug(val_seqs_raw,   mu, sigma, augment=False)
        test_seqs_n = apply_norm_aug(test_seqs,      mu, sigma, augment=False)

        train_ds = SequenceDataset(train_seqs,  train_labels)
        val_ds   = SequenceDataset(val_seqs,    val_labels)
        test_ds  = SequenceDataset(test_seqs_n, test_labels)

        def make_train_dl(bs):
            return DataLoader(
                train_ds, batch_size=bs, shuffle=True,
                collate_fn=collate_fn, num_workers=DL_WORKERS,
                pin_memory=pin,
                persistent_workers=(DL_WORKERS > 0),
                prefetch_factor=2 if DL_WORKERS > 0 else None,
            )

        def make_val_dl(bs):
            return DataLoader(
                val_ds, batch_size=bs, shuffle=False,
                collate_fn=collate_fn, num_workers=DL_WORKERS,
                pin_memory=pin,
                persistent_workers=(DL_WORKERS > 0),
                prefetch_factor=2 if DL_WORKERS > 0 else None,
            )

        test_dl = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=DL_WORKERS,
            pin_memory=pin,
            persistent_workers=(DL_WORKERS > 0),
            prefetch_factor=2 if DL_WORKERS > 0 else None,
        )

        input_size = train_seqs[0].shape[1]
        model      = LightweightMV2(input_size=input_size, num_classes=NUM_CLASSES)

        if s_idx == resume_from:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Model     : {MODEL_NAME}")
            print(f"  Input dim : {input_size}")
            print(f"  Params    : {n_params:,}")

        def make_optimizer(m):
            return torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)

        def make_scheduler(opt):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=args.lr * 0.01)

        model, device, used_env, used_batch, train_dl, val_dl = _do_train(
            model, make_train_dl, make_val_dl, make_optimizer, make_scheduler,
            criterion, args, device_str, args.batch_size, use_early_stopping=True,
        )
        print(f"  Trained on {used_env.upper()} | batch_size={used_batch}")

        model.to(device)
        _, test_acc, y_pred, y_true = evaluate(model, test_dl, criterion, device)
        print(f"  Test accuracy: {test_acc * 100:.2f}%")

        # Shut down DataLoader workers to free file descriptors
        for dl in [train_dl, val_dl, test_dl]:
            dl._iterator = None
        del train_dl, val_dl, test_dl

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

        if args.save_cm:
            safe_id = "".join(c if c.isalnum() else "_" for c in sid)
            save_confusion_matrix(
                conf_mat, CLASS_NAMES,
                title=f"CM – {sid}  [{MODEL_NAME}]",
                path=str(cm_dir / f"CM_Fold_{s_idx+1:02d}_{safe_id}.png"),
            )

        # Accumulate Excel data in memory (written once after all folds)
        sheet = f"Subject_{s_idx+1}"
        all_cm_data[sheet] = pd.DataFrame(conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        all_mt_data[sheet] = pd.DataFrame({
            "Precision": metrics["precision"],
            "Recall":    metrics["recall"],
            "F1Score":   metrics["f1"],
            "MCC":       metrics["mcc"],
        }, index=CLASS_NAMES)

        # Checkpoint after every completed fold
        save_checkpoint(ckpt_path, s_idx, fold_metrics, all_cm_data, all_mt_data)

    # ── Write Excel ONCE after all folds ─────────────────────────────────────
    if all_cm_data:
        excel_cm = out_dir / "LOSO_ConfusionMatrices.xlsx"
        excel_mt = out_dir / "LOSO_PerClassMetrics.xlsx"
        with pd.ExcelWriter(excel_cm, engine="openpyxl") as w:
            for sheet, df in all_cm_data.items():
                df.to_excel(w, sheet_name=sheet)
        with pd.ExcelWriter(excel_mt, engine="openpyxl") as w:
            for sheet, df in all_mt_data.items():
                df.to_excel(w, sheet_name=sheet)
        print(f"  Excel saved → {out_dir}/")

    # ── Summary ──────────────────────────────────────────────────────────────
    loso_summary = {}
    if fold_metrics:
        print(f"\n{'='*60}")
        print(f"  LOSO Summary — {MODEL_NAME} — {len(fold_metrics)} folds")
        print(f"{'='*60}")
        for k in ["accuracy", "precision", "recall", "f1", "mcc"]:
            vals = [m[k] for m in fold_metrics]
            avg, std = np.mean(vals), np.std(vals)
            print(f"  Avg {k:10s}: {avg:.4f}  (std {std:.4f})")
            loso_summary[k] = {"mean": float(avg), "std": float(std)}
        pd.DataFrame(fold_metrics).to_csv(
            out_dir / "LOSO_Summary.csv", index=False)
        print(f"  Summary → {args.out_dir}/LOSO_Summary.csv")

        # Delete checkpoint once all folds are complete
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"  [Checkpoint] deleted — all folds complete.")

    return fold_metrics, loso_summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — FULL RETRAIN ON ALL DATA
# ─────────────────────────────────────────────────────────────────────────────

def retrain_on_all_data(subjects, args):
    """
    Retrain on every subject and every sample.
    No val split, no early stopping — train for exactly args.epochs.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    pin        = device_str == "cuda"

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Full retrain on ALL {len(subjects)} subjects")
    print(f"  Epochs: {args.epochs}  |  LR: {args.lr}  |  "
          f"Batch: {args.batch_size}")
    print(f"{'='*60}")

    all_seqs, all_labels = [], []
    for subj in subjects:
        for smp in subj["samples"]:
            seq = extract_sequence(smp)
            if seq is not None:
                all_seqs.append(seq)
                all_labels.append(smp["label"] - 1)

    print(f"  Total samples: {len(all_seqs)}")

    mu, sigma = compute_normalization(all_seqs)
    norm_seqs = apply_norm_aug(all_seqs, mu, sigma, augment=True)

    input_size = norm_seqs[0].shape[1]
    full_ds    = SequenceDataset(norm_seqs, all_labels)

    def make_full_dl(bs):
        return DataLoader(
            full_ds, batch_size=bs, shuffle=True,
            collate_fn=collate_fn, num_workers=DL_WORKERS,
            pin_memory=pin,
            persistent_workers=(DL_WORKERS > 0),
            prefetch_factor=2 if DL_WORKERS > 0 else None,
        )

    def make_no_val_dl(bs):
        return None

    model    = LightweightMV2(input_size=input_size, num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Input dim: {input_size}  |  Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def make_optimizer(m):
        return torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)

    def make_scheduler(opt):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    model, device, used_env, used_batch, _, _ = _do_train(
        model, make_full_dl, make_no_val_dl,
        make_optimizer, make_scheduler,
        criterion, args, device_str, args.batch_size,
        use_early_stopping=False,
    )
    print(f"\n  Full retrain done on {used_env.upper()} | batch_size={used_batch}")
    return model.cpu(), input_size, mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_model(model, input_size, export_dir, mu, sigma, loso_summary):
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    # ── Unwrap torch.compile() before export ────────────────────────────────
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    model.eval().cpu()

    # ── 1. model_state_dict.pt ──────────────────────────────────────────────
    ckpt_path = Path(export_dir) / "model_state_dict.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size":       input_size,
        "num_classes":      NUM_CLASSES,
        "class_names":      CLASS_NAMES,
        "mu":               mu,
        "sigma":            sigma,
        "loso_summary":     loso_summary,
        "training_note": (
            "Weights trained on ALL subjects after LOSO evaluation. "
            "loso_summary contains the unbiased cross-validated metrics."
        ),
    }, ckpt_path)
    print(f"  [Export] state dict    → {ckpt_path}")

    # ── 2. model.onnx ───────────────────────────────────────────────────────
    dummy     = torch.randn(1, 100, input_size)
    onnx_path = Path(export_dir) / "model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input":  {0: "batch_size", 1: "seq_len"},
                      "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"  [Export] ONNX model    → {onnx_path}")

    # ── 3. model_class.py ───────────────────────────────────────────────────
    (Path(export_dir) / "model_class.py").write_text('''\
"""
model_class.py  —  Self-contained LightweightMV2 definition.
Copy this file wherever you need to load model_state_dict.pt.
"""
import torch
import torch.nn as nn


class _InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid          = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1),
            nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, mid, 3, stride, 1, groups=mid),
            nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, out_ch, 1),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        return (x + self.conv(x)) if self.use_res else self.conv(x)


class LightweightMV2(nn.Module):
    """
    Input  : (B, T, C)  — batch x time_steps x features
    Output : (B, num_classes) — raw logits
    """
    def __init__(self, input_size: int, num_classes: int = 24, dropout: float = 0.2):
        super().__init__()
        cfg = [(32,1,1,1),(48,2,6,2),(64,1,6,3),(96,2,6,2),(128,1,6,1)]
        layers = [nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU6(True))]
        in_ch = 32
        for out_ch, s, e, n in cfg:
            for i in range(n):
                layers.append(_InvertedResidual1D(
                    in_ch, out_ch, stride=s if i == 0 else 1, expand=e))
                in_ch = out_ch
        layers += [nn.Conv1d(in_ch, 256, 1), nn.BatchNorm1d(256), nn.ReLU6(True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths=None, padding_mask=None):
        return self.head(self.features(x.permute(0, 2, 1)))
''', encoding="utf-8")
    print(f"  [Export] model_class   → {export_dir}/model_class.py")

    # ── 4. load_and_test.py ─────────────────────────────────────────────────
    (Path(export_dir) / "load_and_test.py").write_text(f'''\
"""
load_and_test.py
================
Loads model_state_dict.pt, runs a dummy forward pass, prints predicted class.
Usage:  python load_and_test.py
"""
import torch, numpy as np
from pathlib import Path
from model_class import LightweightMV2

CHECKPOINT  = Path(__file__).parent / "model_state_dict.pt"
NUM_CLASSES = {NUM_CLASSES}
CLASS_NAMES = {CLASS_NAMES}


def preprocess(raw: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
               downsample: int = 2) -> torch.Tensor:
    """raw: (features, time) → normalised (1, T, features) tensor."""
    seq = raw[:, ::downsample].T.astype(np.float32)
    seq = (seq - mu) / sigma
    return torch.from_numpy(seq).unsqueeze(0)


def main():
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    print(f"Input size    : {{ckpt['input_size']}} features")
    print(f"Training note : {{ckpt.get('training_note', '')}}")
    print()
    print("LOSO cross-validation results (reported metrics):")
    for metric, vals in ckpt.get("loso_summary", {{}}).items():
        print(f"  {{metric:12s}}: mean={{vals['mean']:.4f}}  std={{vals['std']:.4f}}")

    model = LightweightMV2(input_size=ckpt["input_size"], num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("\\nModel loaded successfully.")

    raw       = np.random.randn(ckpt["input_size"], 300).astype(np.float32)
    tensor_in = preprocess(raw, ckpt["mu"], ckpt["sigma"])
    print(f"Input shape : {{tensor_in.shape}}")

    with torch.no_grad():
        logits = model(tensor_in)
        probs  = torch.softmax(logits, dim=-1)
        pred   = probs.argmax().item()

    print(f"Predicted   : {{pred}} — {{CLASS_NAMES[pred]}}")
    print(f"Confidence  : {{probs[0, pred].item():.4f}}")
    print("\\nDummy forward pass PASSED.")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
    print(f"  [Export] load_and_test → {export_dir}/load_and_test.py")

    # ── 5. README.md ────────────────────────────────────────────────────────
    loso_table = "\n".join(
        f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |"
        for k, v in loso_summary.items()
    ) if loso_summary else "| — | — | — |"

    (Path(export_dir) / "README.md").write_text(f"""\
# LightweightMV2 — BFRB / Gesture Classification

## Training Strategy
**Phase 1 — LOSO cross-validation** across all 20 subjects → unbiased performance estimates.  
**Phase 2 — Full retrain on all subjects + all samples** using the same hyperparameters → exported model.

The LOSO metrics below are the reported numbers. The exported weights reflect training on the complete dataset.

## LOSO Cross-Validation Results
| Metric | Mean | Std |
|--------|------|-----|
{loso_table}

## Files
| File | Description |
|------|-------------|
| `model.onnx` | ONNX export (opset 17), dynamic batch & sequence length |
| `model_state_dict.pt` | PyTorch checkpoint — weights + normalisation stats + LOSO summary |
| `model_class.py` | Self-contained model definition, no other project files needed |
| `load_and_test.py` | Loads checkpoint, runs dummy forward pass end-to-end |
| `README.md` | This file |

## Input / Output
| | Shape | Description |
|-|-------|-------------|
| Input  | `(B, T, {input_size})` | Batch × time steps × features |
| Output | `(B, 24)` | Raw logits — apply `softmax` for probabilities |

## Preprocessing (must match training exactly)
1. Raw array shape: `(features={input_size}, time_steps)`
2. **Downsample** → `[:, ::2]`
3. **Transpose** → `(T', features)`
4. **Normalise** → `(x − mu) / sigma`  — stats stored in checkpoint under `"mu"` / `"sigma"`
5. **Batch dim** → `(1, T', {input_size})`

## Classes
| Index | Name |
|-------|------|
{chr(10).join(f"| {i} | {n} |" for i, n in enumerate(CLASS_NAMES))}

## PyTorch Inference
```python
import torch, numpy as np
from model_class import LightweightMV2

ckpt  = torch.load("model_state_dict.pt", map_location="cpu")
model = LightweightMV2(input_size=ckpt["input_size"], num_classes=24)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

raw = np.random.randn(ckpt["input_size"], 300).astype(np.float32)
seq = (raw[:, ::2].T - ckpt["mu"]) / ckpt["sigma"]
x   = torch.from_numpy(seq).unsqueeze(0)   # (1, T, features)

with torch.no_grad():
    pred = torch.softmax(model(x), dim=-1).argmax().item()
print(ckpt["class_names"][pred])
```

## ONNX Inference
```python
import onnxruntime as ort, numpy as np, torch

ckpt    = torch.load("model_state_dict.pt", map_location="cpu")
session = ort.InferenceSession("model.onnx")

raw    = np.random.randn(ckpt["input_size"], 300).astype(np.float32)
seq    = (raw[:, ::2].T - ckpt["mu"]) / ckpt["sigma"]
logits = session.run(["logits"], {{"input": seq[np.newaxis]}})[0]
pred   = logits.argmax(axis=-1)[0]
print(ckpt["class_names"][pred])
```
""", encoding="utf-8")
    print(f"  [Export] README        → {export_dir}/README.md")
    print(f"\n  Export complete → {export_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LightweightMV2: LOSO evaluation → full retrain → export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",         default="subjects_cleaned_20subjects.pkl")
    p.add_argument("--out_dir",      default=OUTPUT_DIR)
    p.add_argument("--export_dir",   default=EXPORT_DIR)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=20,
                   help="Early-stopping patience (LOSO folds only)")
    p.add_argument("--log_interval", type=int,   default=10)
    p.add_argument("--max_folds",    type=int,   default=0,
                   help="Limit LOSO folds for quick testing (0 = all subjects)")
    p.add_argument("--skip_loso",    action="store_true",
                   help="Skip LOSO and go straight to full retrain + export")
    p.add_argument("--save_cm",      action="store_true",
                   help="Save per-fold confusion matrix PNGs (slower)")
    p.add_argument("--compile",      action="store_true",
                   help="Apply torch.compile() — PyTorch 2+ recommended")
    p.add_argument("--workers",      type=int,   default=DL_WORKERS,
                   help="DataLoader worker processes (0=single-process, safest)")
    return p.parse_args()


def main():
    args = parse_args()

    global DL_WORKERS
    DL_WORKERS = args.workers

    if not Path(args.data).exists():
        print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)

    import pickle
    with open(args.data, "rb") as f:
        subjects = pickle.load(f)
    print(f"Loaded {len(subjects)} subjects from {args.data}")

    t0           = time.time()
    loso_summary = {}

    # ── Phase 1: LOSO (evaluation) ───────────────────────────────────────────
    if not args.skip_loso:
        _, loso_summary = run_loso(subjects, args)
    else:
        print("\n[Skipping LOSO — --skip_loso flag set]")

    # ── Phase 2: Full retrain (deployment model) ─────────────────────────────
    final_model, input_size, mu, sigma = retrain_on_all_data(subjects, args)

    # ── Export ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Exporting final model")
    print(f"{'='*60}")
    export_model(final_model, input_size, args.export_dir, mu, sigma, loso_summary)

    print(f"\nTotal time: {(time.time() - t0) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
