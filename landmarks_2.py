import cv2
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import mediapipe as mp

# =============================================================================
# INPUT / OUTPUT ROOTS
# =============================================================================

INPUT_ROOT = Path(r"D:\set 2 bfrb vids")
OUTPUT_ROOT = Path(r"D:\BFRB Coordinates")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# =============================================================================
# CLASS NAMES
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
# MEDIAPIPE
# =============================================================================

mp_holistic = mp.solutions.holistic

POSE_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner",
    "right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]

HAND_NAMES = [
    "WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
]

FACE_FROM_POSE = list(range(0, 11))  # 0–10 only

# =============================================================================
# HELPERS
# =============================================================================

def lm_to_xyz(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float32)

# =============================================================================
# VIDEO PROCESSING (FULL PIPELINE)
# =============================================================================

def process_video(video_path, subject_id, activity_idx):

    class_name = CLASSES.get(activity_idx, f"Class_{activity_idx}")
    out_dir = OUTPUT_ROOT / class_name / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Cannot open:", video_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    body_rows, face_rows, lh_rows, rh_rows, facemesh_rows = [], [], [], [], []

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

            # ---------------- BODY CENTER ----------------
            body_center = None
            if results.pose_landmarks:
                pts = []
                for idx in [11, 12, 23, 24]:  # shoulders + hips
                    lm = results.pose_landmarks.landmark[idx]
                    pts.append(lm_to_xyz(lm, width, height))
                body_center = np.mean(np.vstack(pts), axis=0)

            # ---------------- BODY ----------------
            brow = OrderedDict(frame=frame_idx)
            if results.pose_landmarks and body_center is not None:
                for i in range(11, 33):
                    name = POSE_NAMES[i]
                    xyz = lm_to_xyz(results.pose_landmarks.landmark[i], width, height) - body_center
                    brow[f"{name}_x"] = xyz[0]
                    brow[f"{name}_y"] = xyz[1]
                    brow[f"{name}_z"] = xyz[2]
            else:
                for i in range(11, 33):
                    name = POSE_NAMES[i]
                    brow[f"{name}_x"] = ""
                    brow[f"{name}_y"] = ""
                    brow[f"{name}_z"] = ""
            body_rows.append(brow)

            # ---------------- FACE (FROM POSE) ----------------
            frow = OrderedDict(frame=frame_idx)
            if results.pose_landmarks and body_center is not None:
                for i in FACE_FROM_POSE:
                    name = POSE_NAMES[i]
                    xyz = lm_to_xyz(results.pose_landmarks.landmark[i], width, height) - body_center
                    frow[f"{name}_x"] = xyz[0]
                    frow[f"{name}_y"] = xyz[1]
                    frow[f"{name}_z"] = xyz[2]
            else:
                for i in FACE_FROM_POSE:
                    name = POSE_NAMES[i]
                    frow[f"{name}_x"] = ""
                    frow[f"{name}_y"] = ""
                    frow[f"{name}_z"] = ""
            face_rows.append(frow)

            # ---------------- LEFT HAND ----------------
            lhrow = OrderedDict(frame=frame_idx)
            if results.left_hand_landmarks and body_center is not None:
                for i, name in enumerate(HAND_NAMES):
                    xyz = lm_to_xyz(results.left_hand_landmarks.landmark[i], width, height) - body_center
                    lhrow[f"{name}_x"] = xyz[0]
                    lhrow[f"{name}_y"] = xyz[1]
                    lhrow[f"{name}_z"] = xyz[2]
            else:
                for name in HAND_NAMES:
                    lhrow[f"{name}_x"] = ""
                    lhrow[f"{name}_y"] = ""
                    lhrow[f"{name}_z"] = ""
            lh_rows.append(lhrow)

            # ---------------- RIGHT HAND ----------------
            rhrow = OrderedDict(frame=frame_idx)
            if results.right_hand_landmarks and body_center is not None:
                for i, name in enumerate(HAND_NAMES):
                    xyz = lm_to_xyz(results.right_hand_landmarks.landmark[i], width, height) - body_center
                    rhrow[f"{name}_x"] = xyz[0]
                    rhrow[f"{name}_y"] = xyz[1]
                    rhrow[f"{name}_z"] = xyz[2]
            else:
                for name in HAND_NAMES:
                    rhrow[f"{name}_x"] = ""
                    rhrow[f"{name}_y"] = ""
                    rhrow[f"{name}_z"] = ""
            rh_rows.append(rhrow)

            # ---------------- FACE MESH (468) ----------------
            fmrow = OrderedDict(frame=frame_idx)
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    xyz = lm_to_xyz(lm, width, height)
                    if body_center is not None:
                        xyz = xyz - body_center
                    fmrow[f"fm_{i}_x"] = xyz[0]
                    fmrow[f"fm_{i}_y"] = xyz[1]
                    fmrow[f"fm_{i}_z"] = xyz[2]
            else:
                for i in range(468):
                    fmrow[f"fm_{i}_x"] = ""
                    fmrow[f"fm_{i}_y"] = ""
                    fmrow[f"fm_{i}_z"] = ""
            facemesh_rows.append(fmrow)

            frame_idx += 1

    finally:
        cap.release()
        holistic.close()

    # ---------------- SAVE EXCEL ----------------
    out_path = out_dir / f"{video_path.stem}_coords.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(body_rows).to_excel(writer, sheet_name="Body", index=False)
        pd.DataFrame(face_rows).to_excel(writer, sheet_name="Face", index=False)
        pd.DataFrame(lh_rows).to_excel(writer, sheet_name="LeftHand", index=False)
        pd.DataFrame(rh_rows).to_excel(writer, sheet_name="RightHand", index=False)
        pd.DataFrame(facemesh_rows).to_excel(writer, sheet_name="FaceMesh", index=False)

    print("Saved:", out_path)

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():

    for subject_dir in sorted(INPUT_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        print(f"\n=== Processing Subject {subject_id} ===")

        for act_dir in sorted(subject_dir.iterdir()):
            if not act_dir.is_dir():
                continue

            try:
                activity_idx = int(act_dir.name.split("-")[1])
            except:
                continue

            print(f"\n Activity {activity_idx}: {CLASSES.get(activity_idx,'?')}")

            for video in act_dir.iterdir():
                if video.suffix.lower() in VIDEO_EXTS:
                    print("   Video:", video.name)
                    process_video(video, subject_id, activity_idx)


if __name__ == "__main__":
    main()
