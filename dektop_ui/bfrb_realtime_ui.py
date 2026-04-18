"""
BFRB Real-Time Recognition UI 
================================================
"""

import sys, os, time, collections, threading, importlib.util
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import requests

# ── suppress noise ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["GLOG_minloglevel"]             = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]        = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import warnings; warnings.filterwarnings("ignore")
import logging;  logging.disable(logging.CRITICAL)

import mediapipe as mp

try:
    mp_holistic    = mp.solutions.holistic
    mp_drawing     = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    _USE_LEGACY_MP = True
except AttributeError:
    _USE_LEGACY_MP = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QFileDialog, QFrame, QSizePolicy,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QRect
from PyQt5.QtGui   import QImage, QPixmap, QPainter, QColor, QFont, QBrush

# =============================================================================
# CONSTANTS
# =============================================================================
BUFFER_FRAMES  = 50          # ~5 s at 10 fps
INFER_INTERVAL = 0.5         # seconds between inference calls

CLASS_NAMES = [
    "Cuticle Picking", "Eyeglasses",      "Face Touching",  "Hair Pulling",
    "Hand Waving",     "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting",     "Phone Call",       "Raising Hand",   "Reading",
    "Scratching Arm",  "Sitting Still",    "Sit-to-Stand",   "Standing",
    "Stand-to-Sit",    "Stretching",       "Thumb Sucking",  "Walking",
]

BFRB_CLASSES = {
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting",  "Scratching Arm",
    "Thumb Sucking",
}

# =============================================================================
# MEDIAPIPE LANDMARK NAMES  (mirrors Script 1 exactly)
# =============================================================================
POSE_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
    "index_mcp","index_pip","index_dip","index_tip",
    "middle_mcp","middle_pip","middle_dip","middle_tip",
    "ring_mcp","ring_pip","ring_dip","ring_tip",
    "pinky_mcp","pinky_pip","pinky_dip","pinky_tip",
]

# Body sheet: indices 11-32 of POSE_NAMES
BODY_NAMES = POSE_NAMES[11:]   # 22 landmarks
# Face sheet: indices 0-10
FACE_NAMES = POSE_NAMES[:11]   # 11 landmarks

# =============================================================================
# HARDCODED 275-FEATURE COLUMN ORDER
# (matches the BFRB_Features sheet produced by Script 1)
# =============================================================================
FEATURE_COLS = [
    "head_tilt_angle",
    # right joint angles
    "R_wrist_pinky","R_wrist_thumb","R_wrist_index",
    "R_elbow","R_shoulder",
    "R_ankle_foot","R_ankle_heel","R_knee","R_thumb_index",
    # left joint angles
    "L_wrist_pinky","L_wrist_thumb","L_wrist_index",
    "L_elbow","L_shoulder",
    "L_ankle_foot","L_ankle_heel","L_knee","L_thumb_index",
    # forearm vs axis
    "R_forearm_vs_hip","L_forearm_vs_hip",
    "R_forearm_vs_shoulder","L_forearm_vs_shoulder",
    # wrist distances
    "dist_Rwrist_ear","dist_Rwrist_mouth","dist_Rwrist_eye","dist_Rwrist_nose",
    "dist_Lwrist_ear","dist_Lwrist_mouth","dist_Lwrist_eye","dist_Lwrist_nose",
    "dist_wrist_to_wrist",
    # right hand vectors
    "R_hv171_x","R_hv171_y","R_hv171_z","R_hv171_mag",
    "R_hv172_x","R_hv172_y","R_hv172_z","R_hv172_mag",
    "R_hv171_172_angle",
    # left hand vectors
    "L_hv171_x","L_hv171_y","L_hv171_z","L_hv171_mag",
    "L_hv172_x","L_hv172_y","L_hv172_z","L_hv172_mag",
    "L_hv171_172_angle",
]

# If the model actually expects 275 we need body + face + hand raw coords too.
# Build the full column list exactly as Script 1 does:
#   Body sheet  : body landmark xyz (indices 11-32) → 22*3 = 66 cols
#   Face sheet  : face landmark xyz (indices  0-10) → 11*3 = 33 cols
#   Left_Hand   : hand landmark xyz (0-20)          → 21*3 = 63 cols
#   Right_Hand  : hand landmark xyz (0-20)          → 21*3 = 63 cols
#   BFRB_Features (above without "frame")           →        50 cols
# Total: 66+33+63+63+50 = 275  ✓

def _build_all_cols():
    cols = []
    # Body (22 landmarks, indices 11-32)
    for name in BODY_NAMES:
        for ax in ("x","y","z"):
            cols.append(f"{name}_{ax}")
    # Face (11 landmarks, indices 0-10)
    for name in FACE_NAMES:
        for ax in ("x","y","z"):
            cols.append(f"{name}_{ax}")
    # Left hand (21 landmarks)
    for name in HAND_NAMES:
        for ax in ("x","y","z"):
            cols.append(f"lhand_{name}_{ax}")
    # Right hand (21 landmarks)
    for name in HAND_NAMES:
        for ax in ("x","y","z"):
            cols.append(f"rhand_{name}_{ax}")
    # Derived features (without "frame")
    cols += FEATURE_COLS
    return cols

ALL_COLS = _build_all_cols()
# 22*3 + 11*3 + 21*3 + 21*3 + 50 = 66+33+63+63+50 = 275
assert len(ALL_COLS) == 275, f"Expected 275 cols, got {len(ALL_COLS)}"

# =============================================================================
# GEOMETRY HELPERS  
# =============================================================================

def _angle_between(v1, v2):
    if v1 is None or v2 is None:
        return np.nan
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)))

def _joint_angle(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    return _angle_between(a - b, c - b)

def _euclidean(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return float(np.linalg.norm(p2 - p1))

def _vector_angle(pf, pt, qf, qt):
    if any(v is None for v in [pf, pt, qf, qt]):
        return np.nan
    return _angle_between(pt - pf, qt - qf)

# =============================================================================
# FRAME → FEATURE ROW
# Mirrors Script 1's extract_landmarks_from_video + compute_bfrb_features
# =============================================================================

def process_holistic_results(results):
    """
    Given a MediaPipe Holistic result object, return a flat dict of ALL_COLS
    (275 values per frame) plus a bool indicating whether body was detected.
    
    Body/face landmarks are in body-centred world space (same as Script 1).
    Hand landmarks are in image-normalised space (same as Script 1).
    """
    row = {c: np.nan for c in ALL_COLS}

    # ── Body centre (mean of shoulders + hips in world space) ─────────────
    body_pts  = {}   # name → np.array([x,y,z]) in CENTRED world space
    face_pts  = {}
    lhand_pts = {}
    rhand_pts = {}

    body_center = None
    if results.pose_world_landmarks:
        pts = []
        for idx in [11, 12, 23, 24]:
            lm = results.pose_world_landmarks.landmark[idx]
            pts.append(np.array([lm.x, lm.y, lm.z], dtype=np.float32))
        body_center = np.mean(np.stack(pts), axis=0)

    # ── POSE landmarks → Body (idx 11-32) and Face (idx 0-10) ─────────────
    if results.pose_world_landmarks and body_center is not None:
        for i, name in enumerate(POSE_NAMES):
            lm  = results.pose_world_landmarks.landmark[i]
            xyz = np.array([lm.x, lm.y, lm.z], dtype=np.float32) - body_center
            if i >= 11:   # body sheet
                row[f"{name}_x"] = xyz[0]
                row[f"{name}_y"] = xyz[1]
                row[f"{name}_z"] = xyz[2]
                body_pts[name]   = xyz
            else:          # face sheet
                row[f"{name}_x"] = xyz[0]
                row[f"{name}_y"] = xyz[1]
                row[f"{name}_z"] = xyz[2]
                face_pts[name]   = xyz

    # ── Left hand landmarks (image-normalised, as in Script 1) ────────────
    if results.left_hand_landmarks:
        for j, name in enumerate(HAND_NAMES):
            lm = results.left_hand_landmarks.landmark[j]
            row[f"lhand_{name}_x"] = lm.x
            row[f"lhand_{name}_y"] = lm.y
            row[f"lhand_{name}_z"] = lm.z
            lhand_pts[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    # ── Right hand landmarks (image-normalised, as in Script 1) ───────────
    if results.right_hand_landmarks:
        for j, name in enumerate(HAND_NAMES):
            lm = results.right_hand_landmarks.landmark[j]
            row[f"rhand_{name}_x"] = lm.x
            row[f"rhand_{name}_y"] = lm.y
            row[f"rhand_{name}_z"] = lm.z
            rhand_pts[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    # ── Derived BFRB features ─────────────────────────────────────────────
    def b(n):  return body_pts.get(n)
    def f(n):  return face_pts.get(n)
    def lh(n): return lhand_pts.get(n)
    def rh(n): return rhand_pts.get(n)

    # head tilt
    ls, rs = b("left_shoulder"), b("right_shoulder")
    lhip, rhip, nose = b("left_hip"), b("right_hip"), f("nose")
    if all(v is not None for v in [ls, rs, lhip, rhip, nose]):
        sm = (ls + rs) / 2.0
        hm = (lhip + rhip) / 2.0
        row["head_tilt_angle"] = _angle_between(hm - sm, nose - sm)

    # right joint angles
    row["R_wrist_pinky"] = _joint_angle(b("right_elbow"),    b("right_wrist"),    b("right_pinky"))
    row["R_wrist_thumb"] = _joint_angle(b("right_elbow"),    b("right_wrist"),    b("right_thumb"))
    row["R_wrist_index"] = _joint_angle(b("right_elbow"),    b("right_wrist"),    b("right_index"))
    row["R_elbow"]       = _joint_angle(b("right_shoulder"), b("right_elbow"),    b("right_wrist"))
    row["R_shoulder"]    = _joint_angle(b("right_elbow"),    b("right_shoulder"), b("right_hip"))
    row["R_ankle_foot"]  = _joint_angle(b("right_knee"),     b("right_ankle"),    b("right_foot_index"))
    row["R_ankle_heel"]  = _joint_angle(b("right_knee"),     b("right_ankle"),    b("right_heel"))
    row["R_knee"]        = _joint_angle(b("right_hip"),      b("right_knee"),     b("right_ankle"))
    row["R_thumb_index"] = _joint_angle(b("right_thumb"),    b("right_wrist"),    b("right_index"))

    # left joint angles
    row["L_wrist_pinky"] = _joint_angle(b("left_elbow"),     b("left_wrist"),     b("left_pinky"))
    row["L_wrist_thumb"] = _joint_angle(b("left_elbow"),     b("left_wrist"),     b("left_thumb"))
    row["L_wrist_index"] = _joint_angle(b("left_elbow"),     b("left_wrist"),     b("left_index"))
    row["L_elbow"]       = _joint_angle(b("left_shoulder"),  b("left_elbow"),     b("left_wrist"))
    row["L_shoulder"]    = _joint_angle(b("left_elbow"),     b("left_shoulder"),  b("left_hip"))
    row["L_ankle_foot"]  = _joint_angle(b("left_knee"),      b("left_ankle"),     b("left_foot_index"))
    row["L_ankle_heel"]  = _joint_angle(b("left_knee"),      b("left_ankle"),     b("left_heel"))
    row["L_knee"]        = _joint_angle(b("left_hip"),       b("left_knee"),      b("left_ankle"))
    row["L_thumb_index"] = _joint_angle(b("left_thumb"),     b("left_wrist"),     b("left_index"))

    # forearm vs axis
    row["R_forearm_vs_hip"]      = _vector_angle(b("right_elbow"),b("right_wrist"),b("right_hip"),     b("left_hip"))
    row["L_forearm_vs_hip"]      = _vector_angle(b("left_elbow"), b("left_wrist"), b("left_hip"),      b("right_hip"))
    row["R_forearm_vs_shoulder"] = _vector_angle(b("right_elbow"),b("right_wrist"),b("right_shoulder"),b("left_shoulder"))
    row["L_forearm_vs_shoulder"] = _vector_angle(b("left_elbow"), b("left_wrist"), b("left_shoulder"), b("right_shoulder"))

    # wrist distances
    row["dist_Rwrist_ear"]     = _euclidean(b("right_wrist"), f("right_ear"))
    row["dist_Rwrist_mouth"]   = _euclidean(b("right_wrist"), f("mouth_right"))
    row["dist_Rwrist_eye"]     = _euclidean(b("right_wrist"), f("right_eye"))
    row["dist_Rwrist_nose"]    = _euclidean(b("right_wrist"), f("nose"))
    row["dist_Lwrist_ear"]     = _euclidean(b("left_wrist"),  f("left_ear"))
    row["dist_Lwrist_mouth"]   = _euclidean(b("left_wrist"),  f("mouth_left"))
    row["dist_Lwrist_eye"]     = _euclidean(b("left_wrist"),  f("left_eye"))
    row["dist_Lwrist_nose"]    = _euclidean(b("left_wrist"),  f("nose"))
    row["dist_wrist_to_wrist"] = _euclidean(b("left_wrist"),  b("right_wrist"))

    # hand vectors
    for side, hand_fn in [("R", rh), ("L", lh)]:
        pmcp = hand_fn("pinky_mcp")
        tcmc = hand_fn("thumb_cmc")
        tmcp = hand_fn("thumb_mcp")

        if pmcp is not None and tcmc is not None:
            v = tcmc - pmcp
            row[f"{side}_hv171_x"]   = float(v[0])
            row[f"{side}_hv171_y"]   = float(v[1])
            row[f"{side}_hv171_z"]   = float(v[2])
            row[f"{side}_hv171_mag"] = float(np.linalg.norm(v))

        if pmcp is not None and tmcp is not None:
            v = tmcp - pmcp
            row[f"{side}_hv172_x"]   = float(v[0])
            row[f"{side}_hv172_y"]   = float(v[1])
            row[f"{side}_hv172_z"]   = float(v[2])
            row[f"{side}_hv172_mag"] = float(np.linalg.norm(v))

        if all(v is not None for v in [pmcp, tcmc, tmcp]):
            row[f"{side}_hv171_172_angle"] = _angle_between(tcmc - pmcp, tmcp - pmcp)

    has_body = results.pose_world_landmarks is not None and body_center is not None
    return row, has_body

# =============================================================================
# BUFFER → TENSOR
# Mirrors Script 1's preprocessing exactly:
#   build (features, T) → downsample → transpose → normalise
# =============================================================================

def buffer_to_tensor(frame_rows, mu, sigma, expected_features):
    """
    frame_rows     : list of row dicts (ALL_COLS)
    mu, sigma      : (1, F) or (F,) numpy arrays from checkpoint
    expected_features: int — model input_size

    Returns float32 tensor (1, T', F) or None on failure.
    """
    if len(frame_rows) < 10:
        return None

    df = pd.DataFrame(frame_rows, columns=ALL_COLS)

    # ── Select the columns the model was trained on ───────────────────────
    # Model input_size tells us how many features. It should be 275.
    # If checkpoint stored feature_columns use those; otherwise use ALL_COLS.
    if df.shape[1] != expected_features:
        # Try to pick only the derived BFRB features (50 cols)
        if expected_features == len(FEATURE_COLS):
            df = df[FEATURE_COLS]
        else:
            # Fallback: use first expected_features cols
            df = df.iloc[:, :expected_features]

    # ── Fill NaN: forward-fill then backward-fill (fast, no KNN) ─────────
    df = df.ffill().bfill()

    # If still NaN (entire column was NaN), fill with 0
    df = df.fillna(0.0)

    data = df.values.astype(np.float32)   # shape: (T, F)

    # ── Downsample on TIME axis (::2) — matches Script 2 DOWNSAMPLE=2 ────
    data = data[::2, :]                   # (T', F)

    if data.shape[0] < 5:
        return None

    # ── Normalise ─────────────────────────────────────────────────────────
    mu_    = mu.reshape(1, -1) if mu.ndim == 2 else mu.reshape(1, -1)
    sigma_ = sigma.reshape(1, -1) if sigma.ndim == 2 else sigma.reshape(1, -1)

    if mu_.shape[1] != data.shape[1]:
        print(f"[buffer_to_tensor] shape mismatch: data={data.shape[1]}, "
              f"mu={mu_.shape[1]}. Check feature column count.")
        return None

    norm = (data - mu_) / sigma_          # (T', F)
    return torch.from_numpy(norm).unsqueeze(0)   # (1, T', F)

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model(pt_path: str):
    pt_path = Path(pt_path)
    model_class_path = pt_path.parent / "model_class.py"

    spec = importlib.util.spec_from_file_location("model_class", model_class_path)
    mc   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)

    ckpt     = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    input_sz = ckpt["input_size"]
    model    = mc.LightweightMV2(input_size=input_sz, num_classes=20)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mu    = ckpt["mu"].astype(np.float32)     # stored as (1, F) or (F,)
    sigma = ckpt["sigma"].astype(np.float32)

    print(f"[load_model] input_size={input_sz}  mu.shape={mu.shape}  "
          f"sigma.shape={sigma.shape}")
    print(f"[load_model] ALL_COLS count = {len(ALL_COLS)}")
    if input_sz != len(ALL_COLS):
        print(f"[load_model] WARNING: model expects {input_sz} features but "
              f"ALL_COLS has {len(ALL_COLS)}. Will attempt to select matching cols.")

    return model, input_sz, mu, sigma

# =============================================================================
# CAMERA THREAD
# =============================================================================

class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object)   # (annotated BGR frame, results)
    error       = pyqtSignal(str)

    def __init__(self, cam_idx=0):
        super().__init__()
        self.cam_idx  = cam_idx
        self._running = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.cam_idx}")
            return

        # model_complexity=2 and refine_face_landmarks to match Script 1
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                annotated = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
                    )
                for hand_lms in [results.left_hand_landmarks,
                                  results.right_hand_landmarks]:
                    if hand_lms:
                        mp_drawing.draw_landmarks(
                            annotated, hand_lms,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_draw_styles.get_default_hand_landmarks_style(),
                            mp_draw_styles.get_default_hand_connections_style(),
                        )
                self.frame_ready.emit(annotated, results)
                self.msleep(16)   # ~60 fps cap

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

# =============================================================================
# INFERENCE THREAD
# =============================================================================

class InferenceThread(QThread):
    prediction_ready = pyqtSignal(list, float)  # list of (class_name, prob)

    def __init__(self, model, mu, sigma, input_size):
        super().__init__()
        self.model      = model
        self.mu         = mu
        self.sigma      = sigma
        self.input_size = input_size
        self._buffer    = collections.deque(maxlen=BUFFER_FRAMES)
        self._lock      = threading.Lock()
        self._running   = False
        self._trigger   = threading.Event()
        self.last_capture_time = None

    def push_frame(self, row_dict):
        with self._lock:
            self._buffer.append(row_dict)
            self.last_capture_time = time.time()
        self._trigger.set()

    def run(self):
        self._running = True
        while self._running:
            triggered = self._trigger.wait(timeout=INFER_INTERVAL)
            self._trigger.clear()

            with self._lock:
                buf = list(self._buffer)

            if len(buf) < 10:
                continue

            tensor = buffer_to_tensor(buf, self.mu, self.sigma, self.input_size)
            if tensor is None:
                continue

            try:
                url = "https://bfrb-api.onrender.com/predict"
                payload = {"features": tensor.squeeze(0).numpy().tolist()}  # (T', F) as list of lists
                response = requests.post(url, json=payload, timeout=5)
                latency_ms = (time.time() - self.last_capture_time) * 1000
                print(f"Latency: {latency_ms:.1f} ms")
                print(response.status_code)
                print(response.text)
                result_json = response.json()
                if "predictions" not in result_json:
                    print("API Error:", result_json)
                    continue
                result = [(item["class"], float(item["probability"])) for item in result_json["predictions"]]
                self.prediction_ready.emit(result, latency_ms)

            except Exception as e:
                print(f"[Cloud Inference] error: {e}")
               
                # to add this if local
                # with torch.no_grad():
                #     logits = self.model(tensor)
                #     probs  = F.softmax(logits, dim=-1)[0].numpy()
                # top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
                # result = [(CLASS_NAMES[i], float(p)) for i, p in top5]
                # self.prediction_ready.emit(result)
            #except Exception as e:
                #print(f"[InferenceThread] error: {e}")

            time.sleep(INFER_INTERVAL)

    def stop(self):
        self._running = False
        self._trigger.set()
        self.wait()

# =============================================================================
# CONFIDENCE BAR WIDGET
# =============================================================================

class ConfidenceBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_text = ""
        self.value      = 0.0
        self.is_bfrb    = False
        self.setFixedHeight(46)

    def set_data(self, label, value, is_bfrb):
        self.label_text = label
        self.value      = value
        self.is_bfrb    = is_bfrb
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H    = self.width(), self.height()
        label_w = 190
        bar_x   = label_w + 8
        bar_w   = W - bar_x - 62
        bar_h   = 12
        bar_y   = (H - bar_h) // 2

        p.setPen(QColor(210, 210, 225))
        p.setFont(QFont("Consolas", 10))
        p.drawText(QRect(0, 0, label_w, H), Qt.AlignVCenter | Qt.AlignLeft,
                   self.label_text)

        p.setBrush(QColor(35, 35, 50))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)

        fill_w = max(0, int(bar_w * self.value))
        if fill_w > 0:
            color = QColor(255, 70, 100) if self.is_bfrb else QColor(70, 195, 130)
            p.setBrush(color)
            p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)

        p.setPen(QColor(170, 170, 195))
        p.setFont(QFont("Consolas", 10))
        p.drawText(QRect(bar_x + bar_w + 6, 0, 55, H),
                   Qt.AlignVCenter | Qt.AlignLeft,
                   f"{self.value * 100:.1f}%")

# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFRB Real-Time Recognition")
        self.setMinimumSize(1120, 700)
        self._apply_theme()

        self.model         = None
        self.mu            = None
        self.sigma         = None
        self.input_size    = 0
        self.cam_thread    = None
        self.infer_thread  = None
        self.fps_times     = collections.deque(maxlen=30)
        self.missing_body_counter = 0
        self.body_detected = False

        self._build_ui()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0d0d14; color:#ddd; font-family:Consolas; }
            QLabel { color:#ccc; }
            QPushButton {
                background:#1a1a2e; color:#d0d0f0; border:1px solid #3a3a58;
                border-radius:6px; padding:7px 16px; font-size:12px;
            }
            QPushButton:hover   { background:#24244a; border-color:#5555bb; }
            QPushButton:pressed { background:#14143a; }
            QPushButton:disabled { color:#444; border-color:#252525; }
            QFrame#panel {
                background:#0f0f1e; border:1px solid #1c1c34; border-radius:10px;
            }
            QSlider::groove:horizontal { height:4px; background:#252540; border-radius:2px; }
            QSlider::handle:horizontal {
                width:14px; height:14px; margin:-5px 0;
                background:#5555bb; border-radius:7px;
            }
            QSlider::sub-page:horizontal { background:#4444aa; border-radius:2px; }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12,12,12,12)
        root.setSpacing(12)

        # ── Left: video ───────────────────────────────────────────────────
        left = QFrame(); left.setObjectName("panel")
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(10,10,10,10)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(580,440)
        self.video_label.setStyleSheet("background:#000; border-radius:6px;")
        self.video_label.setText("Load model → Start camera")
        lv.addWidget(self.video_label)

        stat_row = QHBoxLayout()
        self.fps_lbl      = QLabel("FPS: —")
        self.latency_lbl = QLabel("Latency: —")
        self.latency_lbl.setStyleSheet("color:#555; font-size:11px;")
        self.buf_lbl      = QLabel("Buffer: 0/90")
        self.track_lbl    = QLabel("● No tracking")
        for l in [self.fps_lbl, self.buf_lbl, self.track_lbl]:
            l.setStyleSheet("color:#555; font-size:11px;")
        stat_row.addWidget(self.fps_lbl)
        stat_row.addWidget(self.buf_lbl)
        stat_row.addWidget(self.latency_lbl)
        stat_row.addStretch()
        stat_row.addWidget(self.track_lbl)
        lv.addLayout(stat_row)
        root.addWidget(left, stretch=3)

        # ── Right: controls + results ─────────────────────────────────────
        right = QVBoxLayout(); right.setSpacing(10)

        # model panel
        mp_ = QFrame(); mp_.setObjectName("panel")
        mv  = QVBoxLayout(mp_); mv.setContentsMargins(14,12,14,12); mv.setSpacing(8)

        t = QLabel("BFRB Detector")
        t.setStyleSheet("font-size:17px; font-weight:600; color:#8888ee; letter-spacing:1px;")
        mv.addWidget(t)

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color:#666; font-size:11px;")
        mv.addWidget(self.model_status)

        self.load_btn = QPushButton("Load model (.pt)")
        self.load_btn.clicked.connect(self._load_model)
        mv.addWidget(self.load_btn)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1c1c34;"); mv.addWidget(sep)

        cam_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.clicked.connect(self._start_camera)
        self.start_btn.setEnabled(False)
        self.stop_btn  = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self._stop_camera)
        self.stop_btn.setEnabled(False)
        cam_row.addWidget(self.start_btn); cam_row.addWidget(self.stop_btn)
        mv.addLayout(cam_row)

        # buffer slider
        buf_row = QHBoxLayout()
        buf_lbl = QLabel("Buffer:"); buf_lbl.setStyleSheet("color:#666; font-size:11px;")
        self.buf_slider  = QSlider(Qt.Horizontal)
        self.buf_slider.setRange(30, 150); self.buf_slider.setValue(BUFFER_FRAMES)
        self.buf_slider.valueChanged.connect(self._on_buf_changed)
        self.buf_val_lbl = QLabel(f"{BUFFER_FRAMES} fr")
        self.buf_val_lbl.setStyleSheet("color:#666; font-size:11px; min-width:40px;")
        buf_row.addWidget(buf_lbl); buf_row.addWidget(self.buf_slider)
        buf_row.addWidget(self.buf_val_lbl)
        mv.addLayout(buf_row)
        right.addWidget(mp_)

        # prediction panel
        pp = QFrame(); pp.setObjectName("panel")
        pv = QVBoxLayout(pp); pv.setContentsMargins(14,12,14,12); pv.setSpacing(6)

        pt = QLabel("Live Predictions")
        pt.setStyleSheet("font-size:12px; color:#6666aa; margin-bottom:2px;")
        pv.addWidget(pt)

        self.top_label = QLabel("—")
        self.top_label.setStyleSheet(
            "font-size:21px; font-weight:700; color:#fff;"
            "background:#13132a; border-radius:8px; padding:10px 14px;"
        )
        self.top_label.setWordWrap(True)
        pv.addWidget(self.top_label)

        self.badge = QLabel()
        self.badge.setFixedHeight(26)
        self.badge.setAlignment(Qt.AlignCenter)
        self._set_badge(None)
        pv.addWidget(self.badge)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color:#1c1c34;"); pv.addWidget(sep2)

        pv.addWidget(QLabel("Top 5", styleSheet="color:#555; font-size:11px;"))
        self.conf_bars = []
        for _ in range(5):
            bar = ConfidenceBar(); pv.addWidget(bar); self.conf_bars.append(bar)

        right.addWidget(pp)
        right.addStretch()
        root.addLayout(right, stretch=2)

    def _set_badge(self, name):
        if name and name in BFRB_CLASSES:
            self.badge.setText("⚠  BFRB Detected")
            self.badge.setStyleSheet(
                "background:#3a0f1e; color:#ff5577; border:1px solid #771530;"
                "border-radius:5px; font-size:12px; font-weight:600; padding:2px 10px;")
        elif name:
            self.badge.setText("✓  Normal Behaviour")
            self.badge.setStyleSheet(
                "background:#0c2b1c; color:#44cc77; border:1px solid #1a5c38;"
                "border-radius:5px; font-size:12px; font-weight:600; padding:2px 10px;")
        else:
            self.badge.setText("Waiting for inference…")
            self.badge.setStyleSheet(
                "background:#181828; color:#444; border:1px solid #222;"
                "border-radius:5px; font-size:12px; padding:2px 10px;")

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model_state_dict.pt", "", "PyTorch model (*.pt)")
        if not path:
            return
        try:
            self.model, self.input_size, self.mu, self.sigma = load_model(path)
            n = sum(p.numel() for p in self.model.parameters())
            self.model_status.setText(
                f"✓  input={self.input_size}  params={n:,}")
            self.model_status.setStyleSheet("color:#44cc77; font-size:11px;")
            self.start_btn.setEnabled(True)
        except Exception as e:
            self.model_status.setText(f"✗  {e}")
            self.model_status.setStyleSheet("color:#ff5577; font-size:11px;")
            import traceback; traceback.print_exc()

    # ── Camera ────────────────────────────────────────────────────────────
    def _start_camera(self):
        if self.cam_thread and self.cam_thread.isRunning():
            return
        self.cam_thread = CameraThread(cam_idx=0)
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.error.connect(
            lambda e: self.model_status.setText(f"Camera error: {e}"))
        self.cam_thread.start()

        self.infer_thread = InferenceThread(
            self.model, self.mu, self.sigma, self.input_size)
        self.infer_thread.prediction_ready.connect(self._on_prediction)
        self.infer_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_camera(self):
        if self.cam_thread:   self.cam_thread.stop();   self.cam_thread  = None
        if self.infer_thread: self.infer_thread.stop(); self.infer_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_lbl.setText("FPS: —")
        self.buf_lbl.setText("Buffer: 0/90")
        self.track_lbl.setText("● No tracking")
        self.track_lbl.setStyleSheet("color:#555; font-size:11px;")

    # ── Frame handler ─────────────────────────────────────────────────────
    def _on_frame(self, frame, results):
        now = time.time()
        self.fps_times.append(now)
        if len(self.fps_times) > 1:
            fps = (len(self.fps_times)-1) / (self.fps_times[-1]-self.fps_times[0])
            self.fps_lbl.setText(f"FPS: {fps:.1f}")

        row, has_body = process_holistic_results(results)

        has_lhand = results.left_hand_landmarks  is not None
        has_rhand = results.right_hand_landmarks is not None
        full_body_visible = (
            has_body and
            has_lhand and
            has_rhand
        )
        if full_body_visible:
            # reset missing counter
            self.missing_body_counter = 0
            self.body_detected = True

            self.track_lbl.setText(
                "● Full body detected — inference running"
            )
            self.track_lbl.setStyleSheet(
                "color:#44cc77; font-size:11px;"
            )

            if self.infer_thread:
                self.infer_thread.push_frame(row)

                buf_len = len(self.infer_thread._buffer)
                cap_val = self.buf_slider.value()

                self.buf_lbl.setText(
                    f"Buffer: {buf_len}/{cap_val}"
                )

        else:
            # tracking lost
            self.missing_body_counter += 1

            if self.missing_body_counter >= 10:
                self.body_detected = False

                self.track_lbl.setText(
                    "● Please show full body"
                )
                self.track_lbl.setStyleSheet(
                    "color:#ff5555; font-size:11px;"
                )

                if self.infer_thread:
                    self.infer_thread._buffer.clear()

                self.buf_lbl.setText(
                    "Buffer: cleared"
                )

            else:
                self.track_lbl.setText(
                    f"● Lost tracking ({self.missing_body_counter}/10)"
                )
                self.track_lbl.setStyleSheet(
                    "color:#ccaa30; font-size:11px;"
                )

        # ── Display frame on UI ────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qimg = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )

        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(pix)
        # if has_body and has_lhand and has_rhand:
        #     self.track_lbl.setText("● Body + Both Hands")
        #     self.track_lbl.setStyleSheet("color:#44cc77; font-size:11px;")
        #     if self.infer_thread:
        #         self.infer_thread.push_frame(row)
        #         buf_len = len(self.infer_thread._buffer)
        #         cap_val = self.buf_slider.value()
        #         self.buf_lbl.setText(f"Buffer: {buf_len}/{cap_val}")
        # elif has_body:
        #     self.track_lbl.setText("● Body only — show both hands")
        #     self.track_lbl.setStyleSheet("color:#ccaa30; font-size:11px;")
        #     # Still push — hand features will be NaN, ffill will handle gaps
        #     if self.infer_thread:
        #         self.infer_thread.push_frame(row)
        #         buf_len = len(self.infer_thread._buffer)
        #         self.buf_lbl.setText(f"Buffer: {buf_len}/{self.buf_slider.value()}")
        # else:
        #     self.track_lbl.setText("● No body detected")
        #     self.track_lbl.setStyleSheet("color:#cc3344; font-size:11px;")
        #     if self.infer_thread:
        #         self.infer_thread._buffer.clear()
        #     self.buf_lbl.setText("Buffer: cleared")

        # # Display frame
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # h, w, ch = rgb.shape
        # qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        # pix  = QPixmap.fromImage(qimg).scaled(
        #     self.video_label.width(), self.video_label.height(),
        #     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # self.video_label.setPixmap(pix)

    # ── Prediction handler ─────────────────────────────────────────────────
    def _on_prediction(self, top5, latency):
        if not top5:
            return
        top_name, _ = top5[0]
        self.top_label.setText(top_name)
        self._set_badge(top_name)
        self.latency_lbl.setText(f"Latency: {latency:.1f} ms")
        for i, bar in enumerate(self.conf_bars):
            if i < len(top5):
                name, prob = top5[i]
                bar.set_data(name, prob, name in BFRB_CLASSES)
            else:
                bar.set_data("", 0.0, False)

    # ── Buffer slider ──────────────────────────────────────────────────────
    def _on_buf_changed(self, val):
        global BUFFER_FRAMES
        BUFFER_FRAMES = val
        self.buf_val_lbl.setText(f"{val} fr")
        if self.infer_thread:
            self.infer_thread._buffer = collections.deque(
                self.infer_thread._buffer, maxlen=val)

    def closeEvent(self, event):
        self._stop_camera()
        event.accept()

# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
