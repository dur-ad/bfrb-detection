"""
COMPLETE BFRB PIPELINE (Single Progress Bar Version)
1. Extract MediaPipe landmarks from videos
2. Process landmarks (gap detection, KNN imputation)
3. Compute angles and distances for BFRB analysis (including head_tilt_angle)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from sklearn.impute import KNNImputer
import re
from collections import OrderedDict
import mediapipe as mp
import os
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_ROOT   = Path("C:\\Users\\desk-7\\Documents\\test\\input_test")
COORDS_ROOT  = Path("C:\\Users\\desk-7\\Documents\\test\\intermediate_test")
BFRB_ROOT    = Path("C:\\Users\\desk-7\\Documents\\test\\output_test")
DISCARD_ROOT = COORDS_ROOT / "discard"

COORDS_ROOT.mkdir(exist_ok=True, parents=True)
BFRB_ROOT.mkdir(exist_ok=True, parents=True)
DISCARD_ROOT.mkdir(exist_ok=True, parents=True)

LONG_GAP_THRESHOLD = 10
KNN_NEIGHBORS      = 5
VIDEO_EXTS         = (".mp4", ".avi", ".mov", ".mkv")

# =============================================================================
# CLASS NAMES
# =============================================================================
CLASSES = {
    1:  "1. Hair Pulling",
    2:  "2. Nail Biting",
    3:  "3. Nose Picking",
    4:  "4. Thumb Sucking",
    5:  "5. Repetitive Adjustment of Eyeglasses",
    6:  "6. Knuckle Cracking",
    7:  "7. Face Touching",
    8:  "8. Leg Shaking",
    9:  "9. Scratching Arm",
    10: "10. Cuticle Picking",
    11: "11. Leg Scratching",
    12: "12. Phone Call",
    13: "13. Eating",
    14: "14. Drinking",
    15: "15. Stretching",
    16: "16. Hand Waving",
    17: "17. Reading",
    18: "18. Using Phone",
    19: "19. Standing",
    20: "20. Sit-to-Stand",
    21: "21. Stand-to-Sit",
    22: "22. Walking",
    23: "23. Sitting Still",
    24: "24. Raising Hand",
}

# =============================================================================
# MEDIAPIPE SETUP
# =============================================================================
mp_holistic = mp.solutions.holistic if hasattr(mp, "solutions") else None

POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def lm_to_xyz_world(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def get_point_from_body(df, i, name):
    try:
        x = df.loc[i, f"{name}_x"]
        y = df.loc[i, f"{name}_y"]
        z = df.loc[i, f"{name}_z"]
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            return None
        return np.array([float(x), float(y), float(z)])
    except:
        return None

def get_point_from_face(df, i, name):
    try:
        x = df.loc[i, f"{name}_x"]
        y = df.loc[i, f"{name}_y"]
        z = df.loc[i, f"{name}_z"]
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            return None
        return np.array([float(x), float(y), float(z)])
    except:
        return None

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

# =============================================================================
# FEATURE COMPUTATION (angles + distances + head_tilt_angle)
# =============================================================================

def compute_bfrb_features(body_df, face_df):
    """
    Compute all BFRB features per frame:
      - head_tilt_angle
      - Right/Left side joint angles
      - Wrist-to-face distances
      - Wrist-to-wrist distance
    """
    feature_rows = []

    for i in range(len(body_df)):
        row = {"frame": i}

        # ----- HEAD TILT ANGLE -----
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

        # ----- RIGHT SIDE ANGLES -----
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

        # ----- LEFT SIDE ANGLES -----
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

        # ----- DISTANCES -----
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

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)

# =============================================================================
# PART 1: LANDMARK EXTRACTION
# =============================================================================

def extract_landmarks_from_video(video_path, subject_id, activity_idx):
    class_name = CLASSES.get(activity_idx, f"Class_{activity_idx}")
    out_dir = COORDS_ROOT / class_name / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    body_rows, face_rows = [], []

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    try:
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            body_center = None
            if results.pose_world_landmarks:
                pts = []
                for idx in [11, 12, 23, 24]:
                    lm = results.pose_world_landmarks.landmark[idx]
                    pts.append(lm_to_xyz_world(lm))
                body_center = np.mean(np.vstack(pts), axis=0)

            brow = OrderedDict(frame=frame_idx)
            frow = OrderedDict(frame=frame_idx)

            for i in range(33):
                name = POSE_NAMES[i]
                if results.pose_world_landmarks and body_center is not None:
                    xyz = lm_to_xyz_world(
                        results.pose_world_landmarks.landmark[i]
                    ) - body_center
                else:
                    xyz = [np.nan, np.nan, np.nan]

                if i >= 11:
                    brow[f"{name}_x"] = xyz[0]
                    brow[f"{name}_y"] = xyz[1]
                    brow[f"{name}_z"] = xyz[2]
                else:
                    frow[f"{name}_x"] = xyz[0]
                    frow[f"{name}_y"] = xyz[1]
                    frow[f"{name}_z"] = xyz[2]

            body_rows.append(brow)
            face_rows.append(frow)
            frame_idx += 1

    finally:
        cap.release()
        holistic.close()

    out_path = out_dir / f"{video_path.stem}_coords.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(body_rows).to_excel(writer, sheet_name="Body",  index=False)
        pd.DataFrame(face_rows).to_excel(writer, sheet_name="Face",  index=False)

    return out_path

# =============================================================================
# PART 2: LANDMARK PROCESSING + FEATURE COMPUTATION
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
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    df[cols] = imputer.fit_transform(df[cols])
    return df

def process_landmarks_file(coord_file):
    body_df = pd.read_excel(coord_file, sheet_name="Body")
    face_df = pd.read_excel(coord_file, sheet_name="Face")

    # --- Discard files with long tracking gaps ---
    if detect_long_gaps(body_df, face_df):
        rel = coord_file.relative_to(COORDS_ROOT)
        discard_path = DISCARD_ROOT / rel
        discard_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(coord_file), str(discard_path))
        return False

    # --- KNN imputation ---
    body_df = knn_interpolate(body_df)
    face_df = knn_interpolate(face_df)

    # --- Compute BFRB features (angles + distances + head_tilt_angle) ---
    features_df = compute_bfrb_features(body_df, face_df)

    # --- Write output: Body, Face, BFRB_Features ---
    rel = coord_file.relative_to(COORDS_ROOT)
    output_file = BFRB_ROOT / rel
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        body_df.to_excel(writer,     sheet_name="Body",          index=False)
        face_df.to_excel(writer,     sheet_name="Face",          index=False)
        features_df.to_excel(writer, sheet_name="BFRB_Features", index=False)

    return True

# =============================================================================
# PART 3: SINGLE PROGRESS BAR PIPELINE
# =============================================================================

def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split('([0-9]+)', str(path.name))
    ]

def run_complete_pipeline():
    print("=" * 60)
    print("COMPLETE BFRB PIPELINE")
    print("=" * 60)

    # -------- Collect All Videos --------
    all_videos = []
    for subject_dir in sorted(VIDEO_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        for act_dir in sorted(subject_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            try:
                activity_idx = int(act_dir.name.split("-")[1])
            except:
                continue
            for video in act_dir.iterdir():
                if video.suffix.lower() in VIDEO_EXTS:
                    all_videos.append((video, subject_id, activity_idx))

    print(f"Total videos found: {len(all_videos)}\n")

    # -------- STEP 1: Landmark Extraction --------
    extracted_files = []
    with tqdm(total=len(all_videos), desc="Extracting landmarks", unit="video") as pbar:
        for video, subject_id, activity_idx in all_videos:
            class_name = CLASSES.get(activity_idx)
            pbar.set_description(f"Subject {subject_id} | {class_name}")
            coord_file = extract_landmarks_from_video(video, subject_id, activity_idx)
            if coord_file:
                extracted_files.append(coord_file)
            pbar.update(1)

    # -------- STEP 2: Processing + Feature Computation --------
    coord_files = list(COORDS_ROOT.glob("**/*_coords.xlsx"))
    coord_files = [f for f in coord_files if not str(f).startswith(str(DISCARD_ROOT))]

    processed = 0
    discarded = 0

    with tqdm(total=len(coord_files), desc="Processing features", unit="file") as pbar:
        for coord_file in sorted(coord_files, key=natural_sort_key):
            parts = coord_file.parts
            class_name = parts[-3]
            subject_id = parts[-2]
            pbar.set_description(f"Subject {subject_id} | {class_name}")

            success = process_landmarks_file(coord_file)
            if success:
                processed += 1
            else:
                discarded += 1
            pbar.update(1)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Videos processed    : {len(extracted_files)}")
    print(f"BFRB files generated: {processed}")
    print(f"Files discarded     : {discarded}")
    print("=" * 60)

# =============================================================================
if __name__ == "__main__":
    run_complete_pipeline()