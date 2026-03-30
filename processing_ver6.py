"""
BFRB PROCESSING-ONLY SCRIPT
Reads existing coords .xlsx files from COORDS_ROOT (with Body, Face, Left_Hand, Right_Hand sheets),
applies gap detection, KNN imputation, computes BFRB features, and writes output to BFRB_ROOT.

Use this script if extraction already completed and only processing needs to be run.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["GLOG_minloglevel"]             = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]        = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from sklearn.impute import KNNImputer
import re
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — must match your pipeline paths
# =============================================================================
COORDS_ROOT  = Path("C:\\Users\\desk-7\\Documents\\main_process\\intermediate_coords_main")
BFRB_ROOT    = Path("C:\\Users\\desk-7\\Documents\\main_process\\output_coords_main")
DISCARD_ROOT = COORDS_ROOT / "discard"

BFRB_ROOT.mkdir(exist_ok=True, parents=True)
DISCARD_ROOT.mkdir(exist_ok=True, parents=True)

LONG_GAP_THRESHOLD = 10
KNN_NEIGHBORS      = 5

# Hand landmark names (indices 0-20)
HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_point_from_df(df, i, name):
    try:
        x = df.loc[i, f"{name}_x"]
        y = df.loc[i, f"{name}_y"]
        z = df.loc[i, f"{name}_z"]
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            return None
        return np.array([float(x), float(y), float(z)])
    except:
        return None

def get_point_from_body(df, i, name):
    return get_point_from_df(df, i, name)

def get_point_from_face(df, i, name):
    return get_point_from_df(df, i, name)

def get_point_from_hand(df, i, name):
    return get_point_from_df(df, i, name)

def angle_between(v1, v2):
    if v1 is None or v2 is None:
        return np.nan
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle if not np.isnan(angle) else np.nan

def joint_angle(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    return angle_between(a - b, c - b)

def euclidean_distance(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return np.sqrt(np.sum((p2 - p1) ** 2))

def vector_angle(p_from, p_to, q_from, q_to):
    if any(p is None for p in [p_from, p_to, q_from, q_to]):
        return np.nan
    v1 = p_to - p_from
    v2 = q_to - q_from
    return angle_between(v1, v2)

# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_bfrb_features(body_df, face_df, lhand_df, rhand_df):
    feature_rows = []

    for i in range(len(body_df)):
        row = {"frame": i}

        # HEAD TILT ANGLE
        left_shoulder  = get_point_from_body(body_df, i, "left_shoulder")
        right_shoulder = get_point_from_body(body_df, i, "right_shoulder")
        left_hip       = get_point_from_body(body_df, i, "left_hip")
        right_hip      = get_point_from_body(body_df, i, "right_hip")
        nose           = get_point_from_face(face_df, i, "nose")

        if (left_shoulder is not None and right_shoulder is not None
                and left_hip is not None and right_hip is not None
                and nose is not None):
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            hip_mid      = (left_hip + right_hip) / 2.0
            torso_vec    = hip_mid - shoulder_mid
            head_vec     = nose - shoulder_mid
            row["head_tilt_angle"] = angle_between(torso_vec, head_vec)
        else:
            row["head_tilt_angle"] = np.nan

        # RIGHT SIDE JOINT ANGLES
        row["R_wrist_pinky"] = joint_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_pinky"))
        row["R_wrist_thumb"] = joint_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_thumb"))
        row["R_wrist_index"] = joint_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_index"))
        row["R_elbow"] = joint_angle(
            get_point_from_body(body_df, i, "right_shoulder"),
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"))
        row["R_shoulder"] = joint_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_shoulder"),
            get_point_from_body(body_df, i, "right_hip"))
        row["R_ankle_foot"] = joint_angle(
            get_point_from_body(body_df, i, "right_knee"),
            get_point_from_body(body_df, i, "right_ankle"),
            get_point_from_body(body_df, i, "right_foot_index"))
        row["R_ankle_heel"] = joint_angle(
            get_point_from_body(body_df, i, "right_knee"),
            get_point_from_body(body_df, i, "right_ankle"),
            get_point_from_body(body_df, i, "right_heel"))
        row["R_knee"] = joint_angle(
            get_point_from_body(body_df, i, "right_hip"),
            get_point_from_body(body_df, i, "right_knee"),
            get_point_from_body(body_df, i, "right_ankle"))
        row["R_thumb_index"] = joint_angle(
            get_point_from_body(body_df, i, "right_thumb"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_index"))

        # LEFT SIDE JOINT ANGLES
        row["L_wrist_pinky"] = joint_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_pinky"))
        row["L_wrist_thumb"] = joint_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_thumb"))
        row["L_wrist_index"] = joint_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_index"))
        row["L_elbow"] = joint_angle(
            get_point_from_body(body_df, i, "left_shoulder"),
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"))
        row["L_shoulder"] = joint_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_shoulder"),
            get_point_from_body(body_df, i, "left_hip"))
        row["L_ankle_foot"] = joint_angle(
            get_point_from_body(body_df, i, "left_knee"),
            get_point_from_body(body_df, i, "left_ankle"),
            get_point_from_body(body_df, i, "left_foot_index"))
        row["L_ankle_heel"] = joint_angle(
            get_point_from_body(body_df, i, "left_knee"),
            get_point_from_body(body_df, i, "left_ankle"),
            get_point_from_body(body_df, i, "left_heel"))
        row["L_knee"] = joint_angle(
            get_point_from_body(body_df, i, "left_hip"),
            get_point_from_body(body_df, i, "left_knee"),
            get_point_from_body(body_df, i, "left_ankle"))
        row["L_thumb_index"] = joint_angle(
            get_point_from_body(body_df, i, "left_thumb"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_index"))

        # FOREARM VS AXIS ANGLES
        row["R_forearm_vs_hip"] = vector_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_hip"),
            get_point_from_body(body_df, i, "left_hip"))
        row["L_forearm_vs_hip"] = vector_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_hip"),
            get_point_from_body(body_df, i, "right_hip"))
        row["R_forearm_vs_shoulder"] = vector_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_shoulder"),
            get_point_from_body(body_df, i, "left_shoulder"))
        row["L_forearm_vs_shoulder"] = vector_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_shoulder"),
            get_point_from_body(body_df, i, "right_shoulder"))

        # DISTANCES
        row["dist_Rwrist_ear"] = euclidean_distance(
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_face(face_df, i, "right_ear"))
        row["dist_Rwrist_mouth"] = euclidean_distance(
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_face(face_df, i, "mouth_right"))
        row["dist_Rwrist_eye"] = euclidean_distance(
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_face(face_df, i, "right_eye"))
        row["dist_Rwrist_nose"] = euclidean_distance(
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_face(face_df, i, "nose"))
        row["dist_Lwrist_ear"] = euclidean_distance(
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_face(face_df, i, "left_ear"))
        row["dist_Lwrist_mouth"] = euclidean_distance(
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_face(face_df, i, "mouth_left"))
        row["dist_Lwrist_eye"] = euclidean_distance(
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_face(face_df, i, "left_eye"))
        row["dist_Lwrist_nose"] = euclidean_distance(
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_face(face_df, i, "nose"))
        row["dist_wrist_to_wrist"] = euclidean_distance(
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "right_wrist"))

        # HAND VECTORS
        for side, hdf in [("R", rhand_df), ("L", lhand_df)]:
            pinky_mcp = get_point_from_hand(hdf, i, "pinky_mcp")   # landmark 17
            thumb_cmc = get_point_from_hand(hdf, i, "thumb_cmc")   # landmark 1
            thumb_mcp = get_point_from_hand(hdf, i, "thumb_mcp")   # landmark 2

            if pinky_mcp is not None and thumb_cmc is not None:
                v17_1 = thumb_cmc - pinky_mcp
                row[f"{side}_hv171_x"]   = float(v17_1[0])
                row[f"{side}_hv171_y"]   = float(v17_1[1])
                row[f"{side}_hv171_z"]   = float(v17_1[2])
                row[f"{side}_hv171_mag"] = float(np.linalg.norm(v17_1))
            else:
                row[f"{side}_hv171_x"]   = np.nan
                row[f"{side}_hv171_y"]   = np.nan
                row[f"{side}_hv171_z"]   = np.nan
                row[f"{side}_hv171_mag"] = np.nan

            if pinky_mcp is not None and thumb_mcp is not None:
                v17_2 = thumb_mcp - pinky_mcp
                row[f"{side}_hv172_x"]   = float(v17_2[0])
                row[f"{side}_hv172_y"]   = float(v17_2[1])
                row[f"{side}_hv172_z"]   = float(v17_2[2])
                row[f"{side}_hv172_mag"] = float(np.linalg.norm(v17_2))
            else:
                row[f"{side}_hv172_x"]   = np.nan
                row[f"{side}_hv172_y"]   = np.nan
                row[f"{side}_hv172_z"]   = np.nan
                row[f"{side}_hv172_mag"] = np.nan

            if (pinky_mcp is not None and thumb_cmc is not None
                    and thumb_mcp is not None):
                row[f"{side}_hv171_172_angle"] = angle_between(
                    thumb_cmc - pinky_mcp,
                    thumb_mcp - pinky_mcp)
            else:
                row[f"{side}_hv171_172_angle"] = np.nan

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)

# =============================================================================
# PROCESSING HELPERS
# =============================================================================

def detect_long_gaps(body_df, face_df):
    body_missing = body_df.drop(columns=["frame"]).isna().all(axis=1)
    face_missing = face_df.drop(columns=["frame"]).isna().all(axis=1)
    combined = body_missing & face_missing
    count = 0
    for val in combined:
        if val:
            count += 1
            if count >= LONG_GAP_THRESHOLD:
                return True
        else:
            count = 0
    return False

def knn_interpolate(df):
    cols = [c for c in df.columns if c != "frame"]

    # Columns that are entirely NaN cannot be imputed — skip them,
    # keep as NaN, impute only the columns that have at least one value.
    all_nan_cols = [c for c in cols if df[c].isna().all()]
    valid_cols   = [c for c in cols if c not in all_nan_cols]

    if not valid_cols:
        # Entire sheet is NaN (e.g. hand never detected) — nothing to impute
        return df

    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    df[valid_cols] = imputer.fit_transform(df[valid_cols])
    return df

def process_landmarks_file(coord_file):
    body_df  = pd.read_excel(coord_file, sheet_name="Body")
    face_df  = pd.read_excel(coord_file, sheet_name="Face")
    lhand_df = pd.read_excel(coord_file, sheet_name="Left_Hand")
    rhand_df = pd.read_excel(coord_file, sheet_name="Right_Hand")

    # Discard files with long tracking gaps
    if detect_long_gaps(body_df, face_df):
        rel = coord_file.relative_to(COORDS_ROOT)
        discard_path = DISCARD_ROOT / rel
        discard_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(coord_file), str(discard_path))
        return False

    # KNN imputation for all four sheets
    body_df  = knn_interpolate(body_df)
    face_df  = knn_interpolate(face_df)
    lhand_df = knn_interpolate(lhand_df)
    rhand_df = knn_interpolate(rhand_df)

    # Compute BFRB features
    features_df = compute_bfrb_features(body_df, face_df, lhand_df, rhand_df)

    # Write output
    rel = coord_file.relative_to(COORDS_ROOT)
    output_file = BFRB_ROOT / rel
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        body_df.to_excel(writer,     sheet_name="Body",          index=False)
        face_df.to_excel(writer,     sheet_name="Face",          index=False)
        lhand_df.to_excel(writer,    sheet_name="Left_Hand",     index=False)
        rhand_df.to_excel(writer,    sheet_name="Right_Hand",    index=False)
        features_df.to_excel(writer, sheet_name="BFRB_Features", index=False)

    return True

# =============================================================================
# NATURAL SORT
# =============================================================================

def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split('([0-9]+)', str(path.name))
    ]

# =============================================================================
# MAIN
# =============================================================================

def run_processing_only():
    print("=" * 60)
    print("BFRB PROCESSING-ONLY PIPELINE")
    print("=" * 60)

    coord_files = list(COORDS_ROOT.glob("**/*_coords.xlsx"))
    coord_files = [f for f in coord_files
                   if not str(f).startswith(str(DISCARD_ROOT))]
    coord_files = sorted(coord_files, key=natural_sort_key)

    print(f"Coord files found: {len(coord_files)}\n")

    processed = 0
    discarded = 0

    with tqdm(
        total=len(coord_files),
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for coord_file in coord_files:
            parts      = coord_file.parts
            class_name = parts[-3]
            subject_id = parts[-2]
            pbar.set_description(f"[Process] {subject_id} | {class_name:<42}")

            success = process_landmarks_file(coord_file)
            if success:
                processed += 1
            else:
                discarded += 1
            pbar.update(1)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"BFRB files generated: {processed}")
    print(f"Files discarded     : {discarded}")
    print("=" * 60)


if __name__ == "__main__":
    run_processing_only()
