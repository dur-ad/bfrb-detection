"""
load_cell_data.py
=================
Python equivalent of LOSO_LoadCellData.m

Loads multi-sheet Excel files from class/subject folder structure,
concatenates Body + BFRB_Features + Face sheets, and saves a
subjects_per_class.pkl file for use by the LOSO training script.

Usage:
    python load_cell_data.py --root ./  --out subjects_per_class.pkl
"""

import os
import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


CLASS_FOLDERS = [
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

SHEET_NAMES = ["Body", "BFRB_Features", "Face"]


def load_excel_sample(excel_path: Path):
    """
    Read three sheets from one Excel file and return a (features x time) numpy array.
    Returns None if any sheet is missing or unreadable.
    """
    arrays = []
    try:
        for sheet in SHEET_NAMES:
            df = pd.read_excel(excel_path, sheet_name=sheet, header=None)
            # df is (time x features) -> transpose to (features x time)
            arrays.append(df.values.T.astype(np.float32))
    except Exception as e:
        warnings.warn(f"Skipping {excel_path}: {e}")
        return None

    # Align time dimension
    T = min(a.shape[1] for a in arrays)
    arrays = [a[:, :T] for a in arrays]

    # Stack features vertically: (total_features x T)
    return np.concatenate(arrays, axis=0)


def load_all_subjects(root: str, out: str):
    root = Path(root)

    # subjects is a list of dicts:
    #   { 'subjectID': str, 'samples': [ {'data': ndarray, 'label': int, 'class': str} ] }
    subject_map: dict[str, int] = {}
    subjects: list[dict] = []

    for class_idx, class_name in enumerate(tqdm(CLASS_FOLDERS, desc="Classes"), start=1):
        class_path = root / class_name
        if not class_path.is_dir():
            warnings.warn(f"Class folder not found: {class_path}")
            continue

        for subj_dir in sorted(class_path.iterdir()):
            if not subj_dir.is_dir():
                continue
            subject_id = subj_dir.name

            for excel_file in sorted(subj_dir.glob("*.xlsx")):
                data = load_excel_sample(excel_file)
                if data is None:
                    continue

                if subject_id not in subject_map:
                    subject_map[subject_id] = len(subjects)
                    subjects.append({"subjectID": subject_id, "samples": []})

                idx = subject_map[subject_id]
                subjects[idx]["samples"].append(
                    {"data": data, "label": class_idx, "class": class_name}
                )

    print(f"Loaded data for {len(subjects)} subjects.")
    with open(out, "wb") as f:
        pickle.dump(subjects, f)
    print(f"Saved -> {out}")
    return subjects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./", help="Top-level dataset folder")
    parser.add_argument("--out", default="subjects_per_class.pkl")
    args = parser.parse_args()
    load_all_subjects(args.root, args.out)
