"""
BFRB Real-Time Recognition UI  —  Full Application
====================================================
Files needed in same directory:
  - database.py        (included in project)
  - model_class.py     (for local fallback)
  - model_state_dict.pt (optional, for local fallback)

Run:
    python bfrb_realtime_ui.py
"""

import sys, os, time, collections, threading, importlib.util, csv, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import requests

os.environ["TF_CPP_MIN_LOG_LEVEL"]                   = "3"
os.environ["GLOG_minloglevel"]                         = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]                    = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]   = "python"
import warnings; warnings.filterwarnings("ignore")
import logging;  logging.disable(logging.CRITICAL)

import mediapipe as mp
try:
    mp_holistic    = mp.solutions.holistic
    mp_drawing     = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
except AttributeError:
    pass

# ------------------------------------------------------------
# FIX TINY FONTS ON WINDOWS + HIGH-DPI SCREENS
# ------------------------------------------------------------
import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QFrame, QSizePolicy, QLineEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QComboBox,
    QFileDialog, QScrollArea, QMessageBox, QCheckBox, QSpacerItem,
    QGridLayout, QProgressBar,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QRect, QTimer, QSize
from PyQt5.QtGui   import (
    QImage, QPixmap, QPainter, QColor, QFont, QBrush, QPen, QLinearGradient,
    QPalette,
)

import database as db

# =============================================================================
# CONSTANTS
# =============================================================================
BUFFER_FRAMES  = 50
INFER_INTERVAL = 0.5

CLASS_NAMES = [
    "Cuticle Picking", "Eyeglasses",       "Face Touching",  "Hair Pulling",
    "Hand Waving",     "Knuckle Cracking",  "Leg Scratching", "Leg Shaking",
    "Nail Biting",     "Phone Call",        "Raising Hand",   "Reading",
    "Scratching Arm",  "Sitting Still",     "Sit-to-Stand",   "Standing",
    "Stand-to-Sit",    "Stretching",        "Thumb Sucking",  "Walking",
]
BFRB_CLASSES = {
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting", "Scratching Arm", "Thumb Sucking",
}

# =============================================================================
# THEME
# =============================================================================
THEME = {
    "bg":        "#0a0a12",
    "panel":     "#0f0f1e",
    "border":    "#1c1c34",
    "accent":    "#6c63ff",
    "accent2":   "#ff6584",
    "text":      "#e0e0f0",
    "subtext":   "#666688",
    "success":   "#44cc77",
    "warning":   "#ccaa30",
    "danger":    "#ff5577",
    "bfrb_bar":  "#ff4664",
    "norm_bar":  "#46c37b",
}

APP_STYLE = f"""
QMainWindow, QDialog, QWidget {{
    background: {THEME['bg']};
    color: {THEME['text']};
    font-family: 'Segoe UI', Consolas, monospace;
    font-size: 13px;
}}
QLabel {{ color: {THEME['text']}; }}
QLineEdit {{
    background: #13132a;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}}
QLineEdit:focus {{ border-color: {THEME['accent']}; }}
QPushButton {{
    background: #1a1a2e;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
}}
QPushButton:hover   {{ background: #22224a; border-color: {THEME['accent']}; }}
QPushButton:pressed {{ background: #14143a; }}
QPushButton:disabled {{ color: #333355; border-color: #181828; }}
QPushButton#primary {{
    background: {THEME['accent']};
    color: #fff;
    border: none;
    font-weight: 600;
}}
QPushButton#primary:hover   {{ background: #7c74ff; }}
QPushButton#primary:pressed {{ background: #5c54df; }}
QPushButton#danger {{
    background: #3a0f1e;
    color: {THEME['danger']};
    border: 1px solid #551525;
}}
QPushButton#danger:hover {{ background: #4a1428; border-color: {THEME['danger']}; }}
QFrame#panel {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
}}
QTabWidget::pane {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 8px;
}}
QTabBar::tab {{
    background: #0d0d1a;
    color: {THEME['subtext']};
    border: 1px solid {THEME['border']};
    border-bottom: none;
    padding: 9px 22px;
    margin-right: 2px;
    border-radius: 6px 6px 0 0;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background: {THEME['panel']};
    color: {THEME['text']};
    border-color: {THEME['accent']};
}}
QTabBar::tab:hover {{ color: {THEME['text']}; }}
QTableWidget {{
    background: #0d0d1a;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 6px;
    gridline-color: {THEME['border']};
    font-size: 12px;
}}
QTableWidget::item:selected {{ background: #22224a; }}
QHeaderView::section {{
    background: #13132a;
    color: {THEME['subtext']};
    border: none;
    border-bottom: 1px solid {THEME['border']};
    padding: 8px;
    font-size: 12px;
    font-weight: 600;
}}
QSlider::groove:horizontal {{
    height: 4px; background: #252540; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: {THEME['accent']}; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {THEME['accent']}; border-radius: 2px; }}
QComboBox {{
    background: #13132a;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 6px;
    padding: 7px 12px;
}}
QComboBox:focus {{ border-color: {THEME['accent']}; }}
QComboBox QAbstractItemView {{
    background: #13132a;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    selection-background-color: #22224a;
}}
QScrollBar:vertical {{
    background: #0d0d1a; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #252545; border-radius: 4px; min-height: 20px;
}}
QCheckBox {{ color: {THEME['text']}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {THEME['border']}; background: #13132a;
}}
QCheckBox::indicator:checked {{ background: {THEME['accent']}; border-color: {THEME['accent']}; }}
QMessageBox {{ background: {THEME['panel']}; color: {THEME['text']}; }}
"""

def styled_panel():
    f = QFrame(); f.setObjectName("panel"); return f

def section_label(text, color=None):
    l = QLabel(text)
    c = color or THEME['subtext']
    l.setStyleSheet(f"color:{c}; font-size:11px; font-weight:600; letter-spacing:1px;")
    return l

def heading(text, size=16, color=None):
    l = QLabel(text)
    c = color or THEME['text']
    l.setStyleSheet(f"color:{c}; font-size:{size}px; font-weight:700;")
    return l

def hsep():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{THEME['border']};"); return f

# =============================================================================
# MEDIAPIPE + FEATURE EXTRACTION  (unchanged from original)
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
BODY_NAMES = POSE_NAMES[11:]
FACE_NAMES = POSE_NAMES[:11]
FEATURE_COLS = [
    "head_tilt_angle",
    "R_wrist_pinky","R_wrist_thumb","R_wrist_index","R_elbow","R_shoulder",
    "R_ankle_foot","R_ankle_heel","R_knee","R_thumb_index",
    "L_wrist_pinky","L_wrist_thumb","L_wrist_index","L_elbow","L_shoulder",
    "L_ankle_foot","L_ankle_heel","L_knee","L_thumb_index",
    "R_forearm_vs_hip","L_forearm_vs_hip","R_forearm_vs_shoulder","L_forearm_vs_shoulder",
    "dist_Rwrist_ear","dist_Rwrist_mouth","dist_Rwrist_eye","dist_Rwrist_nose",
    "dist_Lwrist_ear","dist_Lwrist_mouth","dist_Lwrist_eye","dist_Lwrist_nose",
    "dist_wrist_to_wrist",
    "R_hv171_x","R_hv171_y","R_hv171_z","R_hv171_mag",
    "R_hv172_x","R_hv172_y","R_hv172_z","R_hv172_mag","R_hv171_172_angle",
    "L_hv171_x","L_hv171_y","L_hv171_z","L_hv171_mag",
    "L_hv172_x","L_hv172_y","L_hv172_z","L_hv172_mag","L_hv171_172_angle",
]

def _build_all_cols():
    cols = []
    for name in BODY_NAMES:
        for ax in ("x","y","z"): cols.append(f"{name}_{ax}")
    for name in FACE_NAMES:
        for ax in ("x","y","z"): cols.append(f"{name}_{ax}")
    for name in HAND_NAMES:
        for ax in ("x","y","z"): cols.append(f"lhand_{name}_{ax}")
    for name in HAND_NAMES:
        for ax in ("x","y","z"): cols.append(f"rhand_{name}_{ax}")
    cols += FEATURE_COLS
    return cols

ALL_COLS = _build_all_cols()
assert len(ALL_COLS) == 275

def _angle_between(v1, v2):
    if v1 is None or v2 is None: return np.nan
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)))

def _joint_angle(a, b, c):
    if a is None or b is None or c is None: return np.nan
    return _angle_between(a - b, c - b)

def _euclidean(p1, p2):
    if p1 is None or p2 is None: return np.nan
    return float(np.linalg.norm(p2 - p1))

def _vector_angle(pf, pt, qf, qt):
    if any(v is None for v in [pf, pt, qf, qt]): return np.nan
    return _angle_between(pt - pf, qt - qf)

def process_holistic_results(results, show_skeleton=True):
    row = {c: np.nan for c in ALL_COLS}
    body_pts = {}; face_pts = {}; lhand_pts = {}; rhand_pts = {}
    body_center = None

    if results.pose_world_landmarks:
        pts = [np.array([results.pose_world_landmarks.landmark[i].x,
                         results.pose_world_landmarks.landmark[i].y,
                         results.pose_world_landmarks.landmark[i].z], dtype=np.float32)
               for i in [11, 12, 23, 24]]
        body_center = np.mean(np.stack(pts), axis=0)

    if results.pose_world_landmarks and body_center is not None:
        for i, name in enumerate(POSE_NAMES):
            lm  = results.pose_world_landmarks.landmark[i]
            xyz = np.array([lm.x, lm.y, lm.z], dtype=np.float32) - body_center
            row[f"{name}_x"] = xyz[0]; row[f"{name}_y"] = xyz[1]; row[f"{name}_z"] = xyz[2]
            if i >= 11: body_pts[name] = xyz
            else:       face_pts[name] = xyz

    if results.left_hand_landmarks:
        for j, name in enumerate(HAND_NAMES):
            lm = results.left_hand_landmarks.landmark[j]
            row[f"lhand_{name}_x"] = lm.x; row[f"lhand_{name}_y"] = lm.y
            row[f"lhand_{name}_z"] = lm.z
            lhand_pts[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    if results.right_hand_landmarks:
        for j, name in enumerate(HAND_NAMES):
            lm = results.right_hand_landmarks.landmark[j]
            row[f"rhand_{name}_x"] = lm.x; row[f"rhand_{name}_y"] = lm.y
            row[f"rhand_{name}_z"] = lm.z
            rhand_pts[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    def b(n): return body_pts.get(n)
    def f(n): return face_pts.get(n)
    def lh(n): return lhand_pts.get(n)
    def rh(n): return rhand_pts.get(n)

    ls, rs = b("left_shoulder"), b("right_shoulder")
    lhip, rhip, nose = b("left_hip"), b("right_hip"), f("nose")
    if all(v is not None for v in [ls, rs, lhip, rhip, nose]):
        sm = (ls + rs) / 2.0; hm = (lhip + rhip) / 2.0
        row["head_tilt_angle"] = _angle_between(hm - sm, nose - sm)

    row["R_wrist_pinky"] = _joint_angle(b("right_elbow"),    b("right_wrist"),    b("right_pinky"))
    row["R_wrist_thumb"] = _joint_angle(b("right_elbow"),    b("right_wrist"),    b("right_thumb"))
    row["R_wrist_index"] = _joint_angle(b("right_elbow"),    b("right_wrist"),    b("right_index"))
    row["R_elbow"]       = _joint_angle(b("right_shoulder"), b("right_elbow"),    b("right_wrist"))
    row["R_shoulder"]    = _joint_angle(b("right_elbow"),    b("right_shoulder"), b("right_hip"))
    row["R_ankle_foot"]  = _joint_angle(b("right_knee"),     b("right_ankle"),    b("right_foot_index"))
    row["R_ankle_heel"]  = _joint_angle(b("right_knee"),     b("right_ankle"),    b("right_heel"))
    row["R_knee"]        = _joint_angle(b("right_hip"),      b("right_knee"),     b("right_ankle"))
    row["R_thumb_index"] = _joint_angle(b("right_thumb"),    b("right_wrist"),    b("right_index"))
    row["L_wrist_pinky"] = _joint_angle(b("left_elbow"),     b("left_wrist"),     b("left_pinky"))
    row["L_wrist_thumb"] = _joint_angle(b("left_elbow"),     b("left_wrist"),     b("left_thumb"))
    row["L_wrist_index"] = _joint_angle(b("left_elbow"),     b("left_wrist"),     b("left_index"))
    row["L_elbow"]       = _joint_angle(b("left_shoulder"),  b("left_elbow"),     b("left_wrist"))
    row["L_shoulder"]    = _joint_angle(b("left_elbow"),     b("left_shoulder"),  b("left_hip"))
    row["L_ankle_foot"]  = _joint_angle(b("left_knee"),      b("left_ankle"),     b("left_foot_index"))
    row["L_ankle_heel"]  = _joint_angle(b("left_knee"),      b("left_ankle"),     b("left_heel"))
    row["L_knee"]        = _joint_angle(b("left_hip"),       b("left_knee"),      b("left_ankle"))
    row["L_thumb_index"] = _joint_angle(b("left_thumb"),     b("left_wrist"),     b("left_index"))
    row["R_forearm_vs_hip"]      = _vector_angle(b("right_elbow"),b("right_wrist"),b("right_hip"),     b("left_hip"))
    row["L_forearm_vs_hip"]      = _vector_angle(b("left_elbow"), b("left_wrist"), b("left_hip"),      b("right_hip"))
    row["R_forearm_vs_shoulder"] = _vector_angle(b("right_elbow"),b("right_wrist"),b("right_shoulder"),b("left_shoulder"))
    row["L_forearm_vs_shoulder"] = _vector_angle(b("left_elbow"), b("left_wrist"), b("left_shoulder"), b("right_shoulder"))
    row["dist_Rwrist_ear"]     = _euclidean(b("right_wrist"), f("right_ear"))
    row["dist_Rwrist_mouth"]   = _euclidean(b("right_wrist"), f("mouth_right"))
    row["dist_Rwrist_eye"]     = _euclidean(b("right_wrist"), f("right_eye"))
    row["dist_Rwrist_nose"]    = _euclidean(b("right_wrist"), f("nose"))
    row["dist_Lwrist_ear"]     = _euclidean(b("left_wrist"),  f("left_ear"))
    row["dist_Lwrist_mouth"]   = _euclidean(b("left_wrist"),  f("mouth_left"))
    row["dist_Lwrist_eye"]     = _euclidean(b("left_wrist"),  f("left_eye"))
    row["dist_Lwrist_nose"]    = _euclidean(b("left_wrist"),  f("nose"))
    row["dist_wrist_to_wrist"] = _euclidean(b("left_wrist"),  b("right_wrist"))

    for side, hand_fn in [("R", rh), ("L", lh)]:
        pmcp = hand_fn("pinky_mcp"); tcmc = hand_fn("thumb_cmc"); tmcp = hand_fn("thumb_mcp")
        if pmcp is not None and tcmc is not None:
            v = tcmc - pmcp
            row[f"{side}_hv171_x"] = float(v[0]); row[f"{side}_hv171_y"] = float(v[1])
            row[f"{side}_hv171_z"] = float(v[2]); row[f"{side}_hv171_mag"] = float(np.linalg.norm(v))
        if pmcp is not None and tmcp is not None:
            v = tmcp - pmcp
            row[f"{side}_hv172_x"] = float(v[0]); row[f"{side}_hv172_y"] = float(v[1])
            row[f"{side}_hv172_z"] = float(v[2]); row[f"{side}_hv172_mag"] = float(np.linalg.norm(v))
        if all(v is not None for v in [pmcp, tcmc, tmcp]):
            row[f"{side}_hv171_172_angle"] = _angle_between(tcmc - pmcp, tmcp - pmcp)

    has_body = results.pose_world_landmarks is not None and body_center is not None
    return row, has_body


def buffer_to_tensor(frame_rows, mu, sigma, expected_features):
    if len(frame_rows) < 10: return None
    df = pd.DataFrame(frame_rows, columns=ALL_COLS)
    if df.shape[1] != expected_features:
        if expected_features == len(FEATURE_COLS): df = df[FEATURE_COLS]
        else: df = df.iloc[:, :expected_features]
    df = df.ffill().bfill().fillna(0.0)
    data = df.values.astype(np.float32)[::2, :]
    if data.shape[0] < 5: return None
    mu_ = mu.reshape(1, -1); sigma_ = sigma.reshape(1, -1)
    if mu_.shape[1] != data.shape[1]: return None
    norm = (data - mu_) / sigma_
    return torch.from_numpy(norm).unsqueeze(0)

# =============================================================================
# CAMERA THREAD
# =============================================================================
class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object)
    error       = pyqtSignal(str)

    def __init__(self, cam_idx=0, show_skeleton=True):
        super().__init__()
        self.cam_idx       = cam_idx
        self.show_skeleton = show_skeleton
        self._running      = False
        self._paused       = False

    def pause(self):  self._paused = True
    def resume(self): self._paused = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.cam_idx}"); return
        with mp_holistic.Holistic(
            static_image_mode=False, model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as holistic:
            while self._running:
                if self._paused:
                    self.msleep(50); continue
                ret, frame = cap.read()
                if not ret: continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True
                annotated = frame.copy()
                if self.show_skeleton and results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style())
                if self.show_skeleton:
                    for hand_lms in [results.left_hand_landmarks, results.right_hand_landmarks]:
                        if hand_lms:
                            mp_drawing.draw_landmarks(
                                annotated, hand_lms, mp_holistic.HAND_CONNECTIONS,
                                mp_draw_styles.get_default_hand_landmarks_style(),
                                mp_draw_styles.get_default_hand_connections_style())
                self.frame_ready.emit(annotated, results)
                self.msleep(16)
        cap.release()

    def stop(self):
        self._running = False; self.wait()

# =============================================================================
# INFERENCE THREAD
# =============================================================================
class InferenceThread(QThread):
    prediction_ready = pyqtSignal(list, float)

    def __init__(self, sensitivity=0.6):
        super().__init__()
        self.sensitivity        = sensitivity
        self._buffer            = collections.deque(maxlen=BUFFER_FRAMES)
        self._lock              = threading.Lock()
        self._running           = False
        self._trigger           = threading.Event()
        self.last_capture_time  = None

    def push_frame(self, row_dict):
        with self._lock:
            self._buffer.append(row_dict)
            self.last_capture_time = time.time()
        self._trigger.set()

    def run(self):
        self._running = True
        while self._running:
            self._trigger.wait(timeout=INFER_INTERVAL)
            self._trigger.clear()
            with self._lock:
                buf = list(self._buffer)
            if len(buf) < 10: continue
            try:
                df = pd.DataFrame(buf, columns=ALL_COLS)
                df = df.ffill().bfill().fillna(0.0)
                data = df.values.astype(np.float32)[::2, :]
                if data.shape[0] < 5: continue
                payload = {"features": data.tolist()}
                t0 = time.time()
                resp = requests.post(
                    "https://bfrb-api.onrender.com/predict",
                    json=payload, timeout=5)
                latency_ms = (time.time() - t0) * 1000
                rj = resp.json()
                if "predictions" not in rj: continue
                result = [(item["class"], float(item["probability"]))
                          for item in rj["predictions"]]
                self.prediction_ready.emit(result, latency_ms)
            except Exception as e:
                print(f"[InferenceThread] {e}")
            time.sleep(INFER_INTERVAL)

    def stop(self):
        self._running = False; self._trigger.set(); self.wait()

# =============================================================================
# CONFIDENCE BAR WIDGET
# =============================================================================
class ConfidenceBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_text = ""; self.value = 0.0; self.is_bfrb = False
        self.setFixedHeight(40)

    def set_data(self, label, value, is_bfrb):
        self.label_text = label; self.value = value; self.is_bfrb = is_bfrb
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        label_w = 180; bar_x = label_w + 8
        bar_w = W - bar_x - 58; bar_h = 10; bar_y = (H - bar_h) // 2
        p.setPen(QColor(THEME['text'])); p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRect(0, 0, label_w, H), Qt.AlignVCenter | Qt.AlignLeft, self.label_text)
        p.setBrush(QColor("#1a1a2e")); p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)
        fill_w = max(0, int(bar_w * self.value))
        if fill_w > 0:
            color = QColor(THEME['bfrb_bar']) if self.is_bfrb else QColor(THEME['norm_bar'])
            p.setBrush(color); p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)
        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRect(bar_x + bar_w + 6, 0, 52, H),
                   Qt.AlignVCenter | Qt.AlignLeft, f"{self.value*100:.1f}%")

# =============================================================================
# MINI CHART WIDGET  (bar chart for session history)
# =============================================================================
class MiniBarChart(QWidget):
    def __init__(self, data: dict, title="", color=THEME['accent'], parent=None):
        super().__init__(parent)
        self.data  = data   # {label: value}
        self.title = title
        self.color = color
        self.setMinimumHeight(160)

    def paintEvent(self, event):
        if not self.data: return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 8, 8, 24, 32
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_t - pad_b
        if chart_w < 10 or chart_h < 10: return

        # title
        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRect(pad_l, 4, W - pad_l, 18), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        items = list(self.data.items())
        mx = max(v for _, v in items) if items else 1
        if mx == 0: mx = 1
        n = len(items)
        bar_w = max(4, chart_w // n - 4)

        for i, (label, val) in enumerate(items):
            x = pad_l + i * (chart_w // n) + (chart_w // n - bar_w) // 2
            bar_h = int(chart_h * val / mx)
            y = pad_t + chart_h - bar_h
            p.setBrush(QColor(self.color)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(x, y, bar_w, bar_h, 3, 3)
            p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
            lbl = label[:8] if len(label) > 8 else label
            p.drawText(QRect(x - 4, H - pad_b + 2, bar_w + 8, pad_b - 2),
                       Qt.AlignHCenter | Qt.AlignTop, lbl)

# =============================================================================
# SHARED DASHBOARD WIDGET  (used by both admin and user)
# =============================================================================
class DashboardWidget(QWidget):
    """
    Shared camera + prediction panel.
    is_admin=True  → shows Start/Pause/Stop + user selector
    is_admin=False → read-only view
    """
    session_started = pyqtSignal(int)   # session_id
    session_ended   = pyqtSignal(int, int)  # session_id, event_count

    def __init__(self, current_user: dict, is_admin=False, parent=None):
        super().__init__(parent)
        self.current_user     = current_user
        self.is_admin         = is_admin
        self.cam_thread       = None
        self.infer_thread     = None
        self.session_id       = None
        self.session_user_id  = None
        self.bfrb_count       = 0
        self.event_count      = 0
        self.last_detection   = None
        self.last_alert_time  = 0
        self.fps_times        = collections.deque(maxlen=30)
        self.missing_counter  = 0
        self._paused          = False
        self.prefs            = db.get_preferences(current_user["user_id"])

        self._build()
        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._update_time_ago)
        self._tick_timer.start(10000)

    # ── Build UI ──────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(12)

        # Left: video
        left = styled_panel(); lv = QVBoxLayout(left); lv.setContentsMargins(10,10,10,10)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 420)
        self.video_label.setStyleSheet("background:#000; border-radius:6px;")
        self.video_label.setText("No active session")
        self.video_label.setAlignment(Qt.AlignCenter)
        lv.addWidget(self.video_label)

        stat_row = QHBoxLayout()
        self.fps_lbl     = QLabel("FPS: —")
        self.latency_lbl = QLabel("Latency: —")
        self.track_lbl   = QLabel("● Idle")
        for l in [self.fps_lbl, self.latency_lbl, self.track_lbl]:
            l.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        stat_row.addWidget(self.fps_lbl)
        stat_row.addWidget(self.latency_lbl)
        stat_row.addStretch()
        stat_row.addWidget(self.track_lbl)
        lv.addLayout(stat_row)
        root.addWidget(left, stretch=3)

        # Right: controls + predictions
        right = QVBoxLayout(); right.setSpacing(10)

        # ── Session control panel (admin only) ────────────────────────────
        if self.is_admin:
            ctrl = styled_panel(); cv = QVBoxLayout(ctrl)
            cv.setContentsMargins(14,12,14,12); cv.setSpacing(8)
            cv.addWidget(heading("Session Control", 14, THEME['accent']))
            cv.addWidget(section_label("SELECT USER"))
            self.user_combo = QComboBox()
            self._refresh_user_combo()
            cv.addWidget(self.user_combo)
            cv.addWidget(hsep())
            btn_row = QHBoxLayout(); btn_row.setSpacing(6)
            self.start_btn = QPushButton("▶  Start")
            self.start_btn.setObjectName("primary")
            self.start_btn.clicked.connect(self._start_session)
            self.pause_btn = QPushButton("⏸  Pause")
            self.pause_btn.clicked.connect(self._pause_session)
            self.pause_btn.setEnabled(False)
            self.stop_btn  = QPushButton("■  Stop")
            self.stop_btn.setObjectName("danger")
            self.stop_btn.clicked.connect(self._stop_session)
            self.stop_btn.setEnabled(False)
            btn_row.addWidget(self.start_btn)
            btn_row.addWidget(self.pause_btn)
            btn_row.addWidget(self.stop_btn)
            cv.addLayout(btn_row)
            self.session_status = QLabel("No session active")
            self.session_status.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
            cv.addWidget(self.session_status)
            right.addWidget(ctrl)

        # ── Stats panel ────────────────────────────────────────────────────
        stats = styled_panel(); sv = QVBoxLayout(stats)
        sv.setContentsMargins(14,12,14,12); sv.setSpacing(8)
        sv.addWidget(heading("Live Stats", 14, THEME['accent']))

        # BFRB counter + last detection
        grid = QGridLayout(); grid.setSpacing(8)
        grid.addWidget(section_label("BFRB EVENTS"), 0, 0)
        grid.addWidget(section_label("LAST DETECTION"), 0, 1)
        self.bfrb_lbl = QLabel("0")
        self.bfrb_lbl.setStyleSheet(
            f"font-size:32px; font-weight:700; color:{THEME['danger']};")
        self.last_det_lbl = QLabel("—")
        self.last_det_lbl.setStyleSheet(
            f"font-size:13px; color:{THEME['text']}; font-weight:600;")
        self.time_ago_lbl = QLabel("")
        self.time_ago_lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        grid.addWidget(self.bfrb_lbl, 1, 0)
        det_col = QVBoxLayout()
        det_col.addWidget(self.last_det_lbl)
        det_col.addWidget(self.time_ago_lbl)
        det_col.setSpacing(2)
        det_w = QWidget(); det_w.setLayout(det_col)
        grid.addWidget(det_w, 1, 1)
        sv.addLayout(grid)
        sv.addWidget(hsep())

        # Badge
        self.badge = QLabel("Waiting for inference…")
        self.badge.setFixedHeight(28)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setStyleSheet(
            f"background:#181828; color:{THEME['subtext']}; border:1px solid {THEME['border']};"
            "border-radius:5px; font-size:12px; padding:2px 10px;")
        sv.addWidget(self.badge)

        # Top prediction
        self.top_label = QLabel("—")
        self.top_label.setStyleSheet(
            f"font-size:20px; font-weight:700; color:#fff;"
            f"background:#13132a; border-radius:8px; padding:10px 14px;")
        self.top_label.setWordWrap(True)
        sv.addWidget(self.top_label)
        sv.addWidget(hsep())
        sv.addWidget(section_label("TOP 5 PREDICTIONS"))
        self.conf_bars = []
        for _ in range(5):
            bar = ConfidenceBar(); sv.addWidget(bar); self.conf_bars.append(bar)
        right.addWidget(stats)
        right.addStretch()
        root.addLayout(right, stretch=2)

    def _refresh_user_combo(self):
        if not self.is_admin: return
        self.user_combo.clear()
        users = db.get_non_admin_users()
        for u in users:
            self.user_combo.addItem(f"{u['username']} ({u['email']})", u['user_id'])
        if not users:
            self.user_combo.addItem("No users available", -1)

    # ── Session control ────────────────────────────────────────────────────
    def _start_session(self):
        uid = self.user_combo.currentData()
        if not uid or uid == -1:
            QMessageBox.warning(self, "No User", "Please select a user first.")
            return
        self.session_user_id = uid
        self.bfrb_count  = 0; self.event_count = 0
        self.bfrb_lbl.setText("0")
        self.last_det_lbl.setText("—"); self.time_ago_lbl.setText("")
        self.last_detection = None

        prefs = db.get_preferences(uid)
        self.prefs = prefs

        self.session_id = db.start_session(uid)
        self.session_started.emit(self.session_id)

        self.cam_thread = CameraThread(
            show_skeleton=bool(prefs.get("skeleton_overlay", 1)))
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.error.connect(lambda e: self.session_status.setText(f"Error: {e}"))
        self.cam_thread.start()

        self.infer_thread = InferenceThread(
            sensitivity=prefs.get("sensitivity", 0.6))
        self.infer_thread.prediction_ready.connect(self._on_prediction)
        self.infer_thread.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.user_combo.setEnabled(False)
        self.session_status.setText(f"● Session active — {datetime.now().strftime('%H:%M:%S')}")
        self.session_status.setStyleSheet(f"color:{THEME['success']}; font-size:11px;")
        self._paused = False

    def _pause_session(self):
        if not self.cam_thread: return
        if self._paused:
            self.cam_thread.resume()
            self.pause_btn.setText("⏸  Pause")
            self.session_status.setText("● Session resumed")
            self.session_status.setStyleSheet(f"color:{THEME['success']}; font-size:11px;")
            self._paused = False
        else:
            self.cam_thread.pause()
            self.pause_btn.setText("▶  Resume")
            self.session_status.setText("⏸  Session paused")
            self.session_status.setStyleSheet(f"color:{THEME['warning']}; font-size:11px;")
            self._paused = True

    def _stop_session(self):
        if self.cam_thread:   self.cam_thread.stop();   self.cam_thread = None
        if self.infer_thread: self.infer_thread.stop(); self.infer_thread = None
        if self.session_id:
            db.end_session(self.session_id, self.event_count)
            self.session_ended.emit(self.session_id, self.event_count)
        self.session_id = None; self.session_user_id = None
        self.video_label.setText("No active session")
        self.video_label.setPixmap(QPixmap())
        self.track_lbl.setText("● Idle")
        self.track_lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        self.fps_lbl.setText("FPS: —"); self.latency_lbl.setText("Latency: —")
        if self.is_admin:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("⏸  Pause")
            self.stop_btn.setEnabled(False)
            self.user_combo.setEnabled(True)
            self.session_status.setText("Session ended")
            self.session_status.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        self._paused = False

    # ── For user-side: receive session signal from admin ──────────────────
    def activate_readonly(self):
        """Called on user side when admin starts a session."""
        pass  # user sees camera feed via same cam_thread (same machine)

    # ── Frame handler ──────────────────────────────────────────────────────
    def _on_frame(self, frame, results):
        now = time.time()
        self.fps_times.append(now)
        if len(self.fps_times) > 1:
            fps = (len(self.fps_times)-1)/(self.fps_times[-1]-self.fps_times[0])
            self.fps_lbl.setText(f"FPS: {fps:.1f}")

        row, has_body = process_holistic_results(results)
        has_lhand = results.left_hand_landmarks is not None
        has_rhand = results.right_hand_landmarks is not None
        full = has_body and has_lhand and has_rhand

        if full:
            self.missing_counter = 0
            self.track_lbl.setText("● Full body — inference running")
            self.track_lbl.setStyleSheet(f"color:{THEME['success']}; font-size:11px;")
            if self.infer_thread: self.infer_thread.push_frame(row)
            if self.infer_thread:
                self.fps_lbl.setText(
                    f"FPS: {(len(self.fps_times)-1)/(self.fps_times[-1]-self.fps_times[0]):.1f}"
                    if len(self.fps_times) > 1 else "FPS: —")
        else:
            self.missing_counter += 1
            if self.missing_counter >= 10:
                self.track_lbl.setText("● Please show full body")
                self.track_lbl.setStyleSheet(f"color:{THEME['danger']}; font-size:11px;")
                if self.infer_thread: self.infer_thread._buffer.clear()
            else:
                self.track_lbl.setText(f"● Lost tracking ({self.missing_counter}/10)")
                self.track_lbl.setStyleSheet(f"color:{THEME['warning']}; font-size:11px;")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    # ── Prediction handler ─────────────────────────────────────────────────
    def _on_prediction(self, top5, latency):
        if not top5: return
        top_name, top_prob = top5[0]
        sensitivity = self.prefs.get("sensitivity", 0.6)
        cooldown    = self.prefs.get("alert_cooldown", 10)

        self.top_label.setText(top_name)
        self.latency_lbl.setText(f"Latency: {latency:.0f}ms")

        if top_name in BFRB_CLASSES and top_prob >= sensitivity:
            self.badge.setText("⚠  BFRB Detected")
            self.badge.setStyleSheet(
                f"background:#3a0f1e; color:{THEME['danger']}; border:1px solid #771530;"
                "border-radius:5px; font-size:12px; font-weight:600; padding:2px 10px;")
            now = time.time()
            if now - self.last_alert_time >= cooldown:
                self.last_alert_time = now
                self.bfrb_count += 1
                self.bfrb_lbl.setText(str(self.bfrb_count))
                self.last_detection = datetime.now()
                self.last_det_lbl.setText(top_name)
                self.time_ago_lbl.setText("just now")
                # log to DB
                if self.session_id:
                    db.log_detection(self.session_id, top_name, top_prob)
                    self.event_count += 1
        else:
            self.badge.setText("✓  Normal Behaviour")
            self.badge.setStyleSheet(
                f"background:#0c2b1c; color:{THEME['success']}; border:1px solid #1a5c38;"
                "border-radius:5px; font-size:12px; font-weight:600; padding:2px 10px;")

        for i, bar in enumerate(self.conf_bars):
            if i < len(top5):
                name, prob = top5[i]
                bar.set_data(name, prob, name in BFRB_CLASSES)
            else:
                bar.set_data("", 0.0, False)

    def _update_time_ago(self):
        if not self.last_detection: return
        delta = datetime.now() - self.last_detection
        mins  = int(delta.total_seconds() // 60)
        if mins == 0:   self.time_ago_lbl.setText("just now")
        elif mins == 1: self.time_ago_lbl.setText("1 min ago")
        else:           self.time_ago_lbl.setText(f"{mins} mins ago")

    def refresh_prefs(self, prefs: dict):
        self.prefs = prefs
        if self.cam_thread:
            self.cam_thread.show_skeleton = bool(prefs.get("skeleton_overlay", 1))

    def cleanup(self):
        self._tick_timer.stop()
        if self.cam_thread:   self.cam_thread.stop()
        if self.infer_thread: self.infer_thread.stop()
        if self.session_id:   db.end_session(self.session_id, self.event_count)

# =============================================================================
# SETTINGS TAB
# =============================================================================
class SettingsTab(QWidget):
    prefs_saved = pyqtSignal(dict)

    def __init__(self, user: dict, parent=None):
        super().__init__(parent)
        self.user  = user
        self.prefs = db.get_preferences(user["user_id"])
        self._build()

    def _build(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)
        outer.addWidget(heading("Settings", 18))
        outer.addWidget(section_label("Adjust detection preferences for your account"))

        panel = styled_panel(); pv = QVBoxLayout(panel)
        pv.setContentsMargins(24, 20, 24, 20); pv.setSpacing(18)

        # Sensitivity
        pv.addWidget(section_label("DETECTION SENSITIVITY"))
        sens_row = QHBoxLayout()
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(50, 90); self.sens_slider.setSingleStep(1)
        self.sens_slider.setValue(int(self.prefs.get("sensitivity", 0.6) * 100))
        self.sens_val = QLabel(f"{self.prefs.get('sensitivity', 0.6):.2f}")
        self.sens_val.setStyleSheet(f"color:{THEME['accent']}; font-weight:700; min-width:40px;")
        self.sens_slider.valueChanged.connect(
            lambda v: self.sens_val.setText(f"{v/100:.2f}"))
        sens_row.addWidget(QLabel("0.50")); sens_row.addWidget(self.sens_slider)
        sens_row.addWidget(QLabel("0.90")); sens_row.addWidget(self.sens_val)
        pv.addLayout(sens_row)
        hint = QLabel("Higher = fewer false positives, but may miss subtle behaviors.")
        hint.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        pv.addWidget(hint)
        pv.addWidget(hsep())

        # Alert cooldown
        pv.addWidget(section_label("ALERT COOLDOWN (SECONDS)"))
        cool_row = QHBoxLayout()
        self.cool_slider = QSlider(Qt.Horizontal)
        self.cool_slider.setRange(5, 60); self.cool_slider.setSingleStep(1)
        self.cool_slider.setValue(int(self.prefs.get("alert_cooldown", 10)))
        self.cool_val = QLabel(f"{self.prefs.get('alert_cooldown', 10)}s")
        self.cool_val.setStyleSheet(f"color:{THEME['accent']}; font-weight:700; min-width:40px;")
        self.cool_slider.valueChanged.connect(
            lambda v: self.cool_val.setText(f"{v}s"))
        cool_row.addWidget(QLabel("5s")); cool_row.addWidget(self.cool_slider)
        cool_row.addWidget(QLabel("60s")); cool_row.addWidget(self.cool_val)
        pv.addLayout(cool_row)
        pv.addWidget(hsep())

        # Skeleton overlay
        pv.addWidget(section_label("SKELETON OVERLAY"))
        self.skel_check = QCheckBox("Show skeleton overlay on camera feed")
        self.skel_check.setChecked(bool(self.prefs.get("skeleton_overlay", 1)))
        pv.addWidget(self.skel_check)
        pv.addWidget(hsep())

        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("primary")
        save_btn.setFixedWidth(160)
        save_btn.clicked.connect(self._save)
        pv.addWidget(save_btn)

        outer.addWidget(panel)
        outer.addStretch()

    def _save(self):
        sensitivity   = self.sens_slider.value() / 100.0
        alert_cooldown = self.cool_slider.value()
        skeleton      = 1 if self.skel_check.isChecked() else 0
        dur = self.prefs.get("session_duration", 60)
        db.save_preferences(self.user["user_id"], sensitivity, alert_cooldown, dur, skeleton)
        self.prefs = db.get_preferences(self.user["user_id"])
        self.prefs_saved.emit(self.prefs)
        QMessageBox.information(self, "Saved", "Settings saved successfully.")

# =============================================================================
# SESSION HISTORY TAB
# =============================================================================
class HistoryTab(QWidget):
    def __init__(self, user: dict, is_admin=False, parent=None):
        super().__init__(parent)
        self.user     = user
        self.is_admin = is_admin
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20); outer.setSpacing(12)

        # Header row
        hdr = QHBoxLayout()
        hdr.addWidget(heading("Session History", 18))
        hdr.addStretch()
        export_csv = QPushButton("⬇  Export CSV")
        export_json = QPushButton("⬇  Export JSON")
        export_csv.clicked.connect(lambda: self._export("csv"))
        export_json.clicked.connect(lambda: self._export("json"))
        hdr.addWidget(export_csv); hdr.addWidget(export_json)
        outer.addLayout(hdr)
        if self.is_admin:
            outer.addWidget(section_label("Showing all users — admin view"))
        else:
            outer.addWidget(section_label("Showing your sessions only"))

        # Sessions table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Session ID", "User" if self.is_admin else "Date",
             "Start Time", "End Time", "Duration", "Events"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            self.table.styleSheet() +
            "QTableWidget { alternate-background-color: #0d0d20; }")
        self.table.verticalHeader().setVisible(False)
        self.table.clicked.connect(self._on_row_click)
        outer.addWidget(self.table)

        # Chart area
        outer.addWidget(section_label("BFRB EVENTS PER SESSION"))
        self.chart_panel = styled_panel()
        self.chart_layout = QVBoxLayout(self.chart_panel)
        self.chart_layout.setContentsMargins(12, 12, 12, 12)
        self.chart_placeholder = QLabel("Select a session to see behavior breakdown")
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setStyleSheet(f"color:{THEME['subtext']};")
        self.chart_layout.addWidget(self.chart_placeholder)
        outer.addWidget(self.chart_panel)

        self.refresh()

    def refresh(self):
        sessions = db.get_all_sessions() if self.is_admin else \
                   db.get_sessions_for_user(self.user["user_id"])
        self.table.setRowCount(len(sessions))
        self._sessions = sessions

        # Build chart data: event_count per session
        chart_data = {}
        for i, s in enumerate(sessions):
            sid = s["session_id"]
            dur = s.get("duration_seconds") or 0
            dur_str = f"{dur//60}m {dur%60}s" if dur else "—"
            end = s.get("end_time", "—") or "—"
            if end != "—":
                try: end = datetime.fromisoformat(end).strftime("%H:%M:%S")
                except: pass
            start = s.get("start_time", "—")
            try: start_fmt = datetime.fromisoformat(start).strftime("%Y-%m-%d %H:%M")
            except: start_fmt = start

            row_data = [
                str(sid),
                s.get("username", start_fmt) if self.is_admin else start_fmt,
                start_fmt,
                end, dur_str,
                str(s.get("event_count", 0)),
            ]
            for j, val in enumerate(row_data):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

            ec = s.get("event_count", 0) or 0
            chart_data[f"#{sid}"] = ec

        # Render overview bar chart
        self._render_chart(chart_data, "BFRB Events per Session", THEME['accent'])

    def _on_row_click(self, index):
        row = index.row()
        if row >= len(self._sessions): return
        session = self._sessions[row]
        events  = db.get_events_for_session(session["session_id"])
        if not events:
            self._render_chart({}, "No events in this session", THEME['subtext']); return
        breakdown = {}
        for e in events:
            bt = e["behavior_type"]
            breakdown[bt] = breakdown.get(bt, 0) + 1
        self._render_chart(breakdown, f"Behavior breakdown — Session #{session['session_id']}",
                           THEME['bfrb_bar'])

    def _render_chart(self, data, title, color):
        for i in reversed(range(self.chart_layout.count())):
            w = self.chart_layout.itemAt(i).widget()
            if w: w.setParent(None)
        if not data:
            lbl = QLabel(title); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color:{THEME['subtext']};")
            self.chart_layout.addWidget(lbl); return
        chart = MiniBarChart(data, title=title, color=color)
        chart.setMinimumHeight(170)
        self.chart_layout.addWidget(chart)

    def _export(self, fmt):
        uid = None if self.is_admin else self.user["user_id"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", f"sessions_export.{fmt}",
            "CSV Files (*.csv)" if fmt == "csv" else "JSON Files (*.json)")
        if not path: return
        try:
            sessions = db.get_all_sessions() if self.is_admin else \
                       db.get_sessions_for_user(self.user["user_id"])
            events   = db._get_all_events() if self.is_admin else \
                       db.get_events_for_user(self.user["user_id"])
            if fmt == "csv":
                with open(path, "w", newline="") as f:
                    if sessions:
                        w = csv.DictWriter(f, fieldnames=sessions[0].keys())
                        w.writeheader(); w.writerows(sessions)
            else:
                with open(path, "w") as f:
                    json.dump({"sessions": sessions, "events": events}, f, indent=2)
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# =============================================================================
# ADMIN USERS TAB
# =============================================================================
class AdminUsersTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20); outer.setSpacing(16)
        outer.addWidget(heading("User Management", 18))

        # Add user form
        form = styled_panel(); fv = QVBoxLayout(form)
        fv.setContentsMargins(20, 16, 20, 16); fv.setSpacing(10)
        fv.addWidget(section_label("ADD NEW USER"))

        row1 = QHBoxLayout()
        self.f_username = QLineEdit(); self.f_username.setPlaceholderText("Username")
        self.f_email    = QLineEdit(); self.f_email.setPlaceholderText("Email")
        row1.addWidget(self.f_username); row1.addWidget(self.f_email)
        fv.addLayout(row1)

        row2 = QHBoxLayout()
        self.f_password = QLineEdit(); self.f_password.setPlaceholderText("Password")
        self.f_password.setEchoMode(QLineEdit.Password)
        self.f_role = QComboBox()
        self.f_role.addItems(["user", "admin"])
        add_btn = QPushButton("Add User")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._add_user)
        row2.addWidget(self.f_password); row2.addWidget(self.f_role)
        row2.addWidget(add_btn)
        fv.addLayout(row2)
        outer.addWidget(form)

        # Users table
        outer.addWidget(section_label("ALL ACCOUNTS"))
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Username", "Email", "Role", "Created", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        outer.addWidget(self.table)
        self.refresh()

    def refresh(self):
        users = db.get_all_users()
        self.table.setRowCount(len(users))
        for i, u in enumerate(users):
            vals = [
                str(u["user_id"]), u["username"], u["email"],
                u["account_type"],
                u["created_at"][:10] if u["created_at"] else "—",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 3:  # role badge color
                    item.setForeground(
                        QColor(THEME['accent']) if v == "admin" else QColor(THEME['success']))
                self.table.setItem(i, j, item)

            del_btn = QPushButton("Delete")
            del_btn.setObjectName("danger")
            uid = u["user_id"]
            del_btn.clicked.connect(lambda _, uid=uid: self._delete_user(uid))
            if u["account_type"] == "admin":
                del_btn.setEnabled(False)
            self.table.setCellWidget(i, 5, del_btn)

    def _add_user(self):
        username = self.f_username.text().strip()
        email    = self.f_email.text().strip()
        password = self.f_password.text()
        role     = self.f_role.currentText()
        if not all([username, email, password]):
            QMessageBox.warning(self, "Missing Fields", "Please fill all fields."); return
        ok, result = db.create_user(username, email, password, role)
        if ok:
            self.f_username.clear(); self.f_email.clear(); self.f_password.clear()
            self.refresh()
            QMessageBox.information(self, "Created", f"User '{username}' created.")
        else:
            QMessageBox.critical(self, "Error", result)

    def _delete_user(self, user_id):
        reply = QMessageBox.question(
            self, "Confirm", "Deactivate this user?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            db.delete_user(user_id); self.refresh()

# =============================================================================
# MAIN WINDOW  (tabbed, role-aware)
# =============================================================================
class AppWindow(QMainWindow):
    def __init__(self, user: dict):
        super().__init__()
        self.user     = user
        self.is_admin = user["account_type"] == "admin"
        self.setWindowTitle(
            f"BFRB Detector  —  {'Admin' if self.is_admin else user['username']}")
        self.setMinimumSize(1180, 740)
        self.setStyleSheet(APP_STYLE)
        self._build()

    def _build(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Top bar ────────────────────────────────────────────────────────
        topbar = QWidget()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(
            f"background:{THEME['panel']}; border-bottom:1px solid {THEME['border']};")
        tb = QHBoxLayout(topbar); tb.setContentsMargins(20, 0, 20, 0)
        logo = QLabel("BFRB Detector")
        logo.setStyleSheet(
            f"color:{THEME['accent']}; font-size:16px; font-weight:700; letter-spacing:2px;")
        role_badge = QLabel("ADMIN" if self.is_admin else "USER")
        role_badge.setStyleSheet(
            f"background:{'#1a1a3a' if self.is_admin else '#0c2b1c'};"
            f"color:{THEME['accent'] if self.is_admin else THEME['success']};"
            f"border:1px solid {THEME['accent'] if self.is_admin else THEME['success']};"
            "border-radius:4px; padding:2px 10px; font-size:11px; font-weight:700;")
        user_lbl = QLabel(f"  {self.user['username']}  ({self.user['email']})")
        user_lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:12px;")
        logout_btn = QPushButton("Log out")
        logout_btn.setFixedWidth(80)
        logout_btn.clicked.connect(self._logout)
        tb.addWidget(logo); tb.addWidget(role_badge); tb.addWidget(user_lbl)
        tb.addStretch(); tb.addWidget(logout_btn)
        root.addWidget(topbar)

        # ── Tabs ───────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setContentsMargins(12, 12, 12, 12)
        content = QWidget(); cl = QVBoxLayout(content)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.addWidget(self.tabs)
        root.addWidget(content)

        # Dashboard
        self.dashboard = DashboardWidget(self.user, is_admin=self.is_admin)
        self.tabs.addTab(self.dashboard, "  Dashboard  ")

        # History
        self.history = HistoryTab(self.user, is_admin=self.is_admin)
        self.tabs.addTab(self.history, "  Session History  ")

        # Settings (user + admin both get settings)
        self.settings_tab = SettingsTab(self.user)
        self.settings_tab.prefs_saved.connect(self.dashboard.refresh_prefs)
        self.tabs.addTab(self.settings_tab, "  Settings  ")

        # Admin-only: Users tab
        if self.is_admin:
            self.users_tab = AdminUsersTab()
            self.tabs.addTab(self.users_tab, "  Users  ")
            # Refresh user combo when users tab changes
            self.tabs.currentChanged.connect(self._on_tab_change)

        # Refresh history when switching to it
        self.tabs.currentChanged.connect(self._on_tab_change)

    def _on_tab_change(self, idx):
        widget = self.tabs.widget(idx)
        if isinstance(widget, HistoryTab):
            widget.refresh()
        if self.is_admin and isinstance(widget, AdminUsersTab):
            widget.refresh()
        if isinstance(widget, DashboardWidget) and self.is_admin:
            self.dashboard._refresh_user_combo()

    def _logout(self):
        self.dashboard.cleanup()
        self.close()
        self._login_window = LoginWindow()
        self._login_window.show()

    def closeEvent(self, event):
        self.dashboard.cleanup()
        event.accept()

# =============================================================================
# LOGIN WINDOW
# =============================================================================
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFRB Detector — Login")
        self.setFixedSize(440, 520)
        self.setStyleSheet(APP_STYLE)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Background panel
        card = QWidget(); card.setFixedSize(440, 520)
        card.setStyleSheet(
            f"background:{THEME['panel']}; border-radius:0px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(50, 50, 50, 50); cl.setSpacing(0)

        # Logo / title
        logo = QLabel("BFRB")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"color:{THEME['accent']}; font-size:42px; font-weight:800; letter-spacing:8px;")
        cl.addWidget(logo)

        sub = QLabel("Real-Time Detection System")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{THEME['subtext']}; font-size:12px; margin-bottom:40px;")
        cl.addWidget(sub)
        cl.addSpacing(32)

        # Email
        email_lbl = section_label("EMAIL ADDRESS")
        cl.addWidget(email_lbl)
        cl.addSpacing(6)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("you@example.com")
        self.email_input.setFixedHeight(42)
        cl.addWidget(self.email_input)
        cl.addSpacing(16)

        # Password
        pass_lbl = section_label("PASSWORD")
        cl.addWidget(pass_lbl)
        cl.addSpacing(6)
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("••••••••")
        self.pass_input.setEchoMode(QLineEdit.Password)
        self.pass_input.setFixedHeight(42)
        self.pass_input.returnPressed.connect(self._login)
        cl.addWidget(self.pass_input)
        cl.addSpacing(28)

        # Login button
        self.login_btn = QPushButton("Sign In")
        self.login_btn.setObjectName("primary")
        self.login_btn.setFixedHeight(44)
        self.login_btn.clicked.connect(self._login)
        cl.addWidget(self.login_btn)
        cl.addSpacing(16)

        # Error label
        self.err_lbl = QLabel("")
        self.err_lbl.setAlignment(Qt.AlignCenter)
        self.err_lbl.setStyleSheet(f"color:{THEME['danger']}; font-size:12px;")
        cl.addWidget(self.err_lbl)
        cl.addStretch()

        note = QLabel("Contact your administrator to create an account.")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        cl.addWidget(note)
        outer.addWidget(card)

    def _login(self):
        email    = self.email_input.text().strip()
        password = self.pass_input.text()
        if not email or not password:
            self.err_lbl.setText("Please enter email and password."); return
        self.login_btn.setEnabled(False)
        self.login_btn.setText("Signing in…")
        user = db.login(email, password)
        if user:
            self._app_window = AppWindow(user)
            self._app_window.show()
            self.close()
        else:
            self.err_lbl.setText("Invalid email or password.")
            self.login_btn.setEnabled(True)
            self.login_btn.setText("Sign In")
            self.pass_input.clear()

# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LoginWindow()
    win.show()
    sys.exit(app.exec_())
