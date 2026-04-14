"""
load_cell_data.py
=================
Loads multi-sheet Excel files from the dataset folder structure and saves
a subjects_per_class.pkl file for use by the LOSO training script.

Folder structure expected:
    ROOT/
        1. Cuticle Picking/
            S2/
                someFile.xlsx   <- has sheets: Body, Face, Left_Hand, Right_Hand, BFRB_Features
            S3/
                ...
        2. Eyeglasses/
            ...

Usage:
    python load_cell_data.py
        (uses DATA_ROOT defined below, or pass --root to override)

    python load_cell_data.py --root "C:/path/to/output_coords_main" --out subjects_per_class.pkl
"""

import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  point at your top-level dataset folder
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\desk-7\Documents\main_process\output_coords_main"


# Sheets to read from each Excel file (order matters — they are stacked vertically)
SHEET_NAMES = ["Body", "Face", "Left_Hand", "Right_Hand", "BFRB_Features"]

# The 20 class names used by the classifier (must match the folder names after the prefix)
CLASS_NAMES = [
    "Cuticle Picking",
    "Eyeglasses",
    "Face Touching",
    "Hair Pulling",
    "Hand Waving",
    "Knuckle Cracking",
    "Leg Scratching",
    "Leg Shaking",
    "Nail Biting",
    "Phone Call",
    "Raising Hand",
    "Reading",
    "Scratching Arm",
    "Sitting Still",
    "Sit-to-Stand",
    "Standing",
    "Stand-to-Sit",
    "Stretching",
    "Thumb Sucking",
    "Walking",
]


def find_class_folder(root: Path, class_name: str) -> Path | None:
    """
    Find a folder whose name ends with the class name, ignoring any numeric prefix.
    e.g. "1. Hair Pulling"  matches  "Hair Pulling"
         "12. Sitting Still" matches  "Sitting Still"
    Returns None if not found.
    """
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        # Strip leading "digits + dot + spaces" prefix then compare
        stripped = folder.name.lstrip("0123456789").lstrip(". ").strip()
        if stripped == class_name:
            return folder
    return None


def load_excel_sample(excel_path: Path) -> np.ndarray | None:
    """
    Read all sheets from one Excel file and return a (total_features × time) numpy array.
    Sheets are stacked vertically in the order defined by SHEET_NAMES.
    Returns None if any sheet is missing or unreadable.
    """
    arrays = []
    try:
        for sheet in SHEET_NAMES:
            df = pd.read_excel(excel_path, sheet_name=sheet, header=0)
            # Remove frame column if present
            df = df.drop(columns=['frame'], errors='ignore')
            arrays.append(df.values.T.astype(np.float32))
    except Exception as e:
        warnings.warn(f"Skipping {excel_path.name}: {e}")
        return None

    # Align all sheets to the shortest time dimension (in case they differ slightly)
    T = min(a.shape[1] for a in arrays)
    arrays = [a[:, :T] for a in arrays]

    # Stack features vertically → (total_features × T)
    return np.concatenate(arrays, axis=0)


def load_all_subjects(root: str, out: str) -> list:
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found:\n  {root}\n"
            "Update DATA_ROOT at the top of load_cell_data.py, "
            "or pass --root on the command line."
        )

    subject_map: dict[str, int] = {}
    subjects: list[dict] = []
    missing_classes = []

    # ── Step 1: Count total Excel files ──────────────────────────────────────
    all_excel_files = []
    for class_name in CLASS_NAMES:
        class_path = find_class_folder(root, class_name)
        if class_path is None:
            continue
        for subj_dir in class_path.iterdir():
            if subj_dir.is_dir():
                all_excel_files.extend(list(subj_dir.glob("*.xlsx")))

    total_files = len(all_excel_files)
    print(f"Total Excel files found: {total_files}")

    # ── Step 2: Global tqdm over all files ───────────────────────────────────
    with tqdm(total=total_files, desc="Processing Excel files") as pbar:

        for class_idx, class_name in enumerate(CLASS_NAMES, start=1):

            class_path = find_class_folder(root, class_name)

            if class_path is None:
                warnings.warn(f"No folder found for class '{class_name}' in {root}")
                missing_classes.append(class_name)
                continue

            subject_dirs = sorted(
                d for d in class_path.iterdir() if d.is_dir()
            )

            if not subject_dirs:
                warnings.warn(f"No subject sub-folders inside: {class_path}")
                continue

            for subj_dir in subject_dirs:
                subject_id = subj_dir.name
                excel_files = sorted(subj_dir.glob("*.xlsx"))

                if not excel_files:
                    warnings.warn(f"No .xlsx files in {subj_dir}")
                    continue

                for excel_file in excel_files:
                    # 🔥 Bonus: show current file name
                    pbar.set_postfix(file=excel_file.name[:25])

                    data = load_excel_sample(excel_file)
                    pbar.update(1)

                    if data is None:
                        continue

                    if subject_id not in subject_map:
                        subject_map[subject_id] = len(subjects)
                        subjects.append({"subjectID": subject_id, "samples": []})

                    idx = subject_map[subject_id]
                    subjects[idx]["samples"].append({
                        "data":  data,
                        "label": class_idx,
                        "class": class_name,
                    })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Subjects loaded : {len(subjects)}")
    total_samples = sum(len(s['samples']) for s in subjects)
    print(f"  Total samples   : {total_samples}")
    if missing_classes:
        print(f"  Missing classes : {missing_classes}")

    print(f"\n  Samples per subject:")
    for s in subjects:
        print(f"    {s['subjectID']:10s}  {len(s['samples'])} samples")
    print(f"{'─'*50}\n")

    with open(out, "wb") as f:
        pickle.dump(subjects, f)

    print(f"Saved → {out}")
    return subjects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load dataset and save subjects_per_class.pkl"
    )
    parser.add_argument(
        "--root",
        default=DATA_ROOT,
        help="Top-level dataset folder (default: DATA_ROOT in script)"
    )
    parser.add_argument(
        "--out",
        default="subjects_per_class.pkl",
        help="Output pickle file name"
    )
    args = parser.parse_args()
    load_all_subjects(args.root, args.out)
