"""
load_cell_data.py
=================
Python replacement for LOSO_LoadCellData.m
Reads Excel files directly — no MATLAB required.

Saves subjects_per_class.pkl which is used by loso_model_zoo.py

Usage
-----
  python load_cell_data.py --data_dir ./

  # If your data is in a specific folder:
  python load_cell_data.py --data_dir /content/data

Output
------
  subjects_per_class.pkl
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match LOSO_LoadCellData.m exactly
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str, output_path: str = "subjects_per_class.pkl"):
    data_dir = Path(data_dir)
    subject_map = {}   # subjectID -> index in subjects list
    subjects    = []   # list of dicts: {subjectID, samples: [...]}

    for class_idx, class_name in enumerate(tqdm(CLASS_FOLDERS, desc="Classes"), start=1):
        class_path = data_dir / class_name

        if not class_path.exists():
            warnings.warn(f"Class folder not found, skipping: {class_path}")
            continue

        # All subject subfolders inside this class folder
        subject_folders = sorted([
            p for p in class_path.iterdir()
            if p.is_dir() and p.name not in {'.', '..'}
        ])

        for subject_folder in subject_folders:
            subject_id = subject_folder.name

            # All .xlsx files for this subject + class
            excel_files = sorted(subject_folder.glob("*.xlsx"))

            for excel_path in excel_files:
                # ── Read the 3 sheets ──────────────────────────────────────
                try:
                    body_data = pd.read_excel(excel_path, sheet_name="Body",
                                              header=None).values.astype(np.float32)
                    bfrb_data = pd.read_excel(excel_path, sheet_name="BFRB_Features",
                                              header=None).values.astype(np.float32)
                    face_data = pd.read_excel(excel_path, sheet_name="Face",
                                              header=None).values.astype(np.float32)
                except Exception as e:
                    warnings.warn(f"Skipping file (cannot read sheets): {excel_path}\n  Reason: {e}")
                    continue

                # ── Transpose to features × time (matches MATLAB bodyData') ─
                body_data = body_data.T
                bfrb_data = bfrb_data.T
                face_data = face_data.T

                # ── Align time lengths ─────────────────────────────────────
                T = min(body_data.shape[1], bfrb_data.shape[1], face_data.shape[1])
                body_data = body_data[:, :T]
                bfrb_data = bfrb_data[:, :T]
                face_data = face_data[:, :T]

                # ── Concatenate: combined features × time ──────────────────
                file_data = np.vstack([body_data, bfrb_data, face_data])

                # ── Store under subject ────────────────────────────────────
                if subject_id not in subject_map:
                    subject_map[subject_id] = len(subjects)
                    subjects.append({
                        "subjectID": subject_id,
                        "samples":   []
                    })

                idx = subject_map[subject_id]
                subjects[idx]["samples"].append({
                    "data":  file_data,      # numpy array: features × time
                    "label": class_idx,      # 1-based (matches MATLAB)
                    "class": class_name,
                })

    print(f"\nLoaded data for {len(subjects)} subjects.")
    for s in subjects:
        print(f"  {s['subjectID']:20s}  →  {len(s['samples'])} samples")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        pickle.dump(subjects, f)
    print(f"\nSaved: {output_path}")
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python replacement for LOSO_LoadCellData.m"
    )
    parser.add_argument(
        "--data_dir", default="./",
        help="Top-level folder containing the 20 class subfolders (default: ./)"
    )
    parser.add_argument(
        "--output", default="subjects_per_class.pkl",
        help="Output pickle file path (default: subjects_per_class.pkl)"
    )
    args = parser.parse_args()

    load_data(args.data_dir, args.output)
