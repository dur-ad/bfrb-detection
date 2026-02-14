"""
COMPLETE BFRB PIPELINE
1. Extract MediaPipe landmarks from videos
2. Process landmarks (gap detection, KNN imputation)
3. Compute angles and distances for BFRB analysis
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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
VIDEO_ROOT = Path(r"D:\process")              # Original video folder
COORDS_ROOT = Path(r"D:\process_coords")       # Intermediate landmarks folder
BFRB_ROOT = Path(r"D:\process_bfrbcoords")     # Final BFRB features folder

# Discard folder in COORDS_ROOT that will mirror the class/subject structure
DISCARD_ROOT = COORDS_ROOT / "discard"          # Base discard folder

# Create directories
COORDS_ROOT.mkdir(exist_ok=True, parents=True)
BFRB_ROOT.mkdir(exist_ok=True, parents=True)
DISCARD_ROOT.mkdir(exist_ok=True, parents=True)

# Processing parameters
LONG_GAP_THRESHOLD = 10    # Max consecutive missing frames
KNN_NEIGHBORS = 5          # For imputation
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# =============================================================================
# CLASS NAMES (from your original code)
# =============================================================================

CLASSES = {
    1: "1. Hair Pulling",
    2: "2. Nail Biting",
    3: "3. Nose Picking",
    4: "4. Thumb Sucking",
    5: "5. Repetitive Adjustment of Eyeglasses",
    6: "6. Knuckle Cracking",
    7: "7. Face Touching",
    8: "8. Leg Shaking",
    9: "9. Scratching Arm",
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
# MEDIAPIPE LANDMARKS
# =============================================================================

mp_holistic = mp.solutions.holistic

POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

# Landmark indices for reference (MediaPipe Pose)
LANDMARK_NAMES = {
    0: "nose", 2: "left_eye", 5: "right_eye",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

# =============================================================================
# PART 1: LANDMARK EXTRACTION (from your original code)
# =============================================================================

def lm_to_xyz_world(lm):
    """Convert MediaPipe landmark to world coordinates"""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def extract_landmarks_from_video(video_path, subject_id, activity_idx):
    """
    Extract MediaPipe landmarks from video and save to Excel
    Returns: Path to saved coordinates file
    """

    class_name = CLASSES.get(activity_idx, f"Class_{activity_idx}")
    out_dir = COORDS_ROOT / class_name / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return None

    body_rows, face_rows = [], []

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Calculate body center (from shoulders and hips)
            body_center = None
            if results.pose_world_landmarks:
                pts = []
                for idx in [11, 12, 23, 24]:  # shoulders and hips
                    lm = results.pose_world_landmarks.landmark[idx]
                    pts.append(lm_to_xyz_world(lm))
                body_center = np.mean(np.vstack(pts), axis=0)

            # Extract BODY landmarks (indices 11-32)
            brow = OrderedDict(frame=frame_idx)
            if results.pose_world_landmarks and body_center is not None:
                for i in range(11, 33):
                    name = POSE_NAMES[i]
                    xyz = lm_to_xyz_world(
                        results.pose_world_landmarks.landmark[i]
                    ) - body_center

                    brow[f"{name}_x"] = xyz[0]
                    brow[f"{name}_y"] = xyz[1]
                    brow[f"{name}_z"] = xyz[2]
            else:
                for i in range(11, 33):
                    name = POSE_NAMES[i]
                    brow[f"{name}_x"] = np.nan
                    brow[f"{name}_y"] = np.nan
                    brow[f"{name}_z"] = np.nan
            body_rows.append(brow)

            # Extract FACE landmarks (indices 0-10)
            frow = OrderedDict(frame=frame_idx)
            if results.pose_world_landmarks and body_center is not None:
                for i in range(0, 11):
                    name = POSE_NAMES[i]
                    xyz = lm_to_xyz_world(
                        results.pose_world_landmarks.landmark[i]
                    ) - body_center

                    frow[f"{name}_x"] = xyz[0]
                    frow[f"{name}_y"] = xyz[1]
                    frow[f"{name}_z"] = xyz[2]
            else:
                for i in range(0, 11):
                    name = POSE_NAMES[i]
                    frow[f"{name}_x"] = np.nan
                    frow[f"{name}_y"] = np.nan
                    frow[f"{name}_z"] = np.nan
            face_rows.append(frow)

            frame_idx += 1

    finally:
        cap.release()
        holistic.close()

    # Save landmarks to Excel
    out_path = out_dir / f"{video_path.stem}_coords.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(body_rows).to_excel(writer, sheet_name="Body", index=False)
        pd.DataFrame(face_rows).to_excel(writer, sheet_name="Face", index=False)

    print(f"  → Landmarks saved: {out_path}")
    return out_path

# =============================================================================
# PART 2: BFRB FEATURE EXTRACTION (adapted from your pipeline)
# =============================================================================

def get_point_from_body(df, i, name):
    """Get 3D point from body dataframe at frame i"""
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
    """Get 3D point from face dataframe at frame i"""
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
    """Calculate angle between two vectors"""
    if v1 is None or v2 is None:
        return np.nan
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle if not np.isnan(angle) else np.nan

def joint_angle(a, b, c):
    """Calculate angle at point b"""
    if a is None or b is None or c is None:
        return np.nan
    return angle_between(a - b, c - b)

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    if p1 is None or p2 is None:
        return np.nan
    diff = p2 - p1
    distance = np.sqrt(np.sum(diff ** 2))
    return distance if not np.isnan(distance) else np.nan

def detect_long_gaps(body_df, face_df):
    """Check for long sequences of missing data in both body and face"""
    body_cols = [c for c in body_df.columns if c != "frame"]
    face_cols = [c for c in face_df.columns if c != "frame"]

    # Check body gaps
    body_missing = body_df[body_cols].isna().all(axis=1)
    # Check face gaps
    face_missing = face_df[face_cols].isna().all(axis=1)

    # Combined missing (both body and face all NaN)
    combined_missing = body_missing & face_missing

    segments = []
    start = None
    for i, m in enumerate(combined_missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segments.append(i - start)
            start = None
    if start is not None:
        segments.append(len(combined_missing) - start)

    return any(gap >= LONG_GAP_THRESHOLD for gap in segments)

def knn_interpolate(df):
    """Impute missing values using KNN for a dataframe"""
    if df.empty:
        return df

    cols = [c for c in df.columns if c != "frame"]
    if not cols:
        return df

    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    df[cols] = imputer.fit_transform(df[cols])
    return df

def compute_bfrb_features(body_df, face_df):
    """
    Compute angles and distances for BFRB analysis
    Using both body and face landmarks
    """
    feature_rows = []

    for i in range(len(body_df)):
        row = {"frame": i}

        # ------------------- RIGHT SIDE ANGLES -------------------
        # Wrist angles
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

        # Elbow angle
        row["R_elbow"] = joint_angle(
            get_point_from_body(body_df, i, "right_shoulder"),
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_wrist"))

        # Shoulder angle
        row["R_shoulder"] = joint_angle(
            get_point_from_body(body_df, i, "right_elbow"),
            get_point_from_body(body_df, i, "right_shoulder"),
            get_point_from_body(body_df, i, "right_hip"))

        # Knee angles
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

        # Thumb-index angle
        row["R_thumb_index"] = joint_angle(
            get_point_from_body(body_df, i, "right_thumb"),
            get_point_from_body(body_df, i, "right_wrist"),
            get_point_from_body(body_df, i, "right_index"))

        # ------------------- LEFT SIDE ANGLES -------------------
        # Wrist angles
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

        # Elbow angle
        row["L_elbow"] = joint_angle(
            get_point_from_body(body_df, i, "left_shoulder"),
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_wrist"))

        # Shoulder angle
        row["L_shoulder"] = joint_angle(
            get_point_from_body(body_df, i, "left_elbow"),
            get_point_from_body(body_df, i, "left_shoulder"),
            get_point_from_body(body_df, i, "left_hip"))

        # Knee angles
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

        # Thumb-index angle
        row["L_thumb_index"] = joint_angle(
            get_point_from_body(body_df, i, "left_thumb"),
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "left_index"))

        # ------------------- DISTANCES (using FACE landmarks) -------------------
        # Right hand to face
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

        # Left hand to face
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

        # Hand to hand
        row["dist_wrist_to_wrist"] = euclidean_distance(
            get_point_from_body(body_df, i, "left_wrist"),
            get_point_from_body(body_df, i, "right_wrist"))

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)

def process_landmarks_file(coord_file):
    """
    Process a landmarks Excel file to generate BFRB features
    """
    print(f"\nProcessing: {coord_file}")

    try:
        # Read both Body and Face sheets
        body_df = pd.read_excel(coord_file, sheet_name="Body")
        face_df = pd.read_excel(coord_file, sheet_name="Face")

        # Check for long gaps in combined data
        if detect_long_gaps(body_df, face_df):
            # Create discard path that mirrors the original structure
            rel_path = coord_file.relative_to(COORDS_ROOT)
            discard_path = DISCARD_ROOT / rel_path
            discard_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file to discard folder while preserving structure
            shutil.move(str(coord_file), str(discard_path))
            print(f" → Discarded due to long gaps → {discard_path}")
            return False

        # KNN imputation for both dataframes
        body_df = knn_interpolate(body_df)
        face_df = knn_interpolate(face_df)

        # Compute BFRB features using both body and face
        features_df = compute_bfrb_features(body_df, face_df)

        # Determine output path in BFRB_ROOT
        rel_path = coord_file.relative_to(COORDS_ROOT)
        output_file = BFRB_ROOT / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save results with ALL three sheets
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            body_df.to_excel(writer, sheet_name="Body", index=False)
            face_df.to_excel(writer, sheet_name="Face", index=False)  # <-- FACE SHEET IS NOW SAVED
            features_df.to_excel(writer, sheet_name="BFRB_Features", index=False)

        print(f" → Processed successfully → {output_file}")
        print(f"    Sheets saved: Body, Face, BFRB_Features")
        return True

    except Exception as e:
        print(f" ✗ FAILED: {e}")
        return False

# =============================================================================
# PART 3: MAIN PIPELINE
# =============================================================================

def natural_sort_key(path):
    """Natural sorting for filenames with numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(path.name))]

def run_complete_pipeline():
    """
    Run the complete pipeline:
    1. Extract landmarks from all videos
    2. Process landmarks to generate BFRB features
    """

    print("=" * 60)
    print("COMPLETE BFRB PIPELINE")
    print("=" * 60)

    # Step 1: Extract landmarks from videos
    print("\n[STEP 1] Extracting landmarks from videos...")
    print("-" * 40)

    extracted_files = []

    for subject_dir in sorted(VIDEO_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        print(f"\nSubject: {subject_id}")

        for act_dir in sorted(subject_dir.iterdir()):
            if not act_dir.is_dir():
                continue

            try:
                activity_idx = int(act_dir.name.split("-")[1])
            except:
                continue

            class_name = CLASSES.get(activity_idx, f"Class_{activity_idx}")
            print(f"  Activity: {class_name}")

            for video in act_dir.iterdir():
                if video.suffix.lower() in VIDEO_EXTS:
                    print(f"    Video: {video.name}")
                    coord_file = extract_landmarks_from_video(
                        video, subject_id, activity_idx
                    )
                    if coord_file:
                        extracted_files.append(coord_file)

    print(f"\n✓ Extracted landmarks from {len(extracted_files)} videos")
    print(f"✓ Landmarks saved in: {COORDS_ROOT}")

    # Step 2: Process landmarks to generate BFRB features
    print("\n[STEP 2] Generating BFRB features...")
    print("-" * 40)

    processed = 0
    discarded = 0

    # Find all extracted coordinate files (excluding those already in discard)
    coord_files = list(COORDS_ROOT.glob("**/*_coords.xlsx"))
    # Exclude files that are already in the discard folder
    coord_files = [f for f in coord_files if not str(f).startswith(str(DISCARD_ROOT))]

    print(f"Found {len(coord_files)} landmark files to process")

    for coord_file in sorted(coord_files, key=natural_sort_key):
        success = process_landmarks_file(coord_file)
        if success:
            processed += 1
        else:
            discarded += 1

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Videos processed: {len(extracted_files)}")
    print(f"Landmark files generated: {len(coord_files)}")
    print(f"BFRB features generated: {processed}")
    print(f"Files discarded: {discarded}")
    print(f"\nFolder structure:")
    print(f"  📁 {VIDEO_ROOT} - Original videos")
    print(f"  📁 {COORDS_ROOT} - Landmark coordinates")
    print(f"      └── Class_Name/")
    print(f"          └── Subject_ID/")
    print(f"              ├── video_coords.xlsx (Body + Face sheets)")
    print(f"  📁 {DISCARD_ROOT} - Discarded files (mirrors structure)")
    print(f"      └── Class_Name/")
    print(f"          └── Subject_ID/")
    print(f"              └── video_coords.xlsx")
    print(f"  📁 {BFRB_ROOT} - Final BFRB features")
    print(f"      └── Class_Name/")
    print(f"          └── Subject_ID/")
    print(f"              └── video_coords.xlsx (Body + Face + BFRB_Features)")
    print("=" * 60)

# =============================================================================
# RUN PIPELINE
# =============================================================================

if __name__ == "__main__":
    run_complete_pipeline()