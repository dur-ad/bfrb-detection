"""
BFRB Real-Time Recognition UI  —  Full Application  (with Metrics Reporting)
=============================================================================
Files needed in same directory:
  - database.py        (included in project)
  - model_class.py     (for local fallback)
  - model_state_dict.pt (optional, for local fallback)

Run:
    python bfrb_realtime_ui.py

Metrics are auto-saved to ./metrics/ after every session as JSON + CSV.
"""

import sys, os, time, collections, threading, importlib.util, csv, json, math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import requests
try:
    import psutil
except ImportError:
    psutil = None

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

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QFrame, QSizePolicy, QLineEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QComboBox,
    QFileDialog, QScrollArea, QMessageBox, QCheckBox, QSpacerItem,
    QGridLayout, QProgressBar, QAbstractItemView,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal, QRect, QTimer, QSize, QPointF
from PyQt5.QtGui   import (
    QImage, QPixmap, QPainter, QColor, QFont, QBrush, QPen, QLinearGradient,
    QPalette, QIcon,
)

import database as db

# =============================================================================
# CONSTANTS
# =============================================================================
WINDOW_SECONDS = 5
TARGET_FPS = 10
BUFFER_FRAMES = WINDOW_SECONDS * TARGET_FPS

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
BFRB_CLASSES = {
    "Cuticle Picking", "Eyeglasses", "Face Touching", "Hair Pulling",
    "Knuckle Cracking", "Leg Scratching", "Leg Shaking",
    "Nail Biting", "Scratching Arm", "Thumb Sucking","Nose Picking"
}

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

# =============================================================================
# THEME
# =============================================================================
THEME = {
    "bg":        "#0b0b18",
    "panel":     "#11112a",
    "border":    "#23234a",
    "accent":    "#7b6fff",
    "accent2":   "#ff6b8a",
    "text":      "#eaeaf8",
    "subtext":   "#7878a8",
    "success":   "#3ddc84",
    "warning":   "#f0b429",
    "danger":    "#ff4d6d",
    "bfrb_bar":  "#ff4d6d",
    "norm_bar":  "#3ddc84",
    "card":      "#16163a",
}

APP_STYLE = f"""
QMainWindow, QDialog, QWidget {{
    background: {THEME['bg']};
    color: {THEME['text']};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 14px;
}}
QLabel {{ color: {THEME['text']}; }}
QLineEdit {{
    background: {THEME['card']};
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 7px;
    padding: 9px 14px;
    font-size: 14px;
}}
QLineEdit:focus {{ border-color: {THEME['accent']}; }}
QPushButton {{
    background: #1e1e40;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 7px;
    padding: 9px 20px;
    font-size: 14px;
}}
QPushButton:hover   {{ background: #28285a; border-color: {THEME['accent']}; }}
QPushButton:pressed {{ background: #161638; }}
QPushButton:disabled {{ color: #33335a; border-color: #1a1a38; }}
QPushButton#primary {{
    background: {THEME['accent']};
    color: #fff;
    border: none;
    font-weight: 700;
    font-size: 14px;
}}
QPushButton#primary:hover   {{ background: #9088ff; }}
QPushButton#primary:pressed {{ background: #5e58d0; }}
QPushButton#danger {{
    background: #3a0f1e;
    color: {THEME['danger']};
    border: 1px solid #661530;
    font-size: 14px;
}}
QPushButton#danger:hover {{ background: #4e1428; border-color: {THEME['danger']}; }}
QFrame#panel {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
}}
QTabWidget::pane {{
    background: {THEME['panel']};
    border: 1px solid {THEME['border']};
    border-radius: 10px;
}}
QTabBar::tab {{
    background: #0e0e22;
    color: {THEME['subtext']};
    border: 1px solid {THEME['border']};
    border-bottom: none;
    padding: 10px 26px;
    margin-right: 3px;
    border-radius: 7px 7px 0 0;
    font-size: 14px;
}}
QTabBar::tab:selected {{
    background: {THEME['panel']};
    color: {THEME['text']};
    border-color: {THEME['accent']};
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {THEME['text']}; }}
QTableWidget {{
    background: #0e0e22;
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 7px;
    gridline-color: {THEME['border']};
    font-size: 13px;
}}
QTableWidget::item:selected {{ background: #28285a; }}
QHeaderView::section {{
    background: {THEME['card']};
    color: {THEME['subtext']};
    border: none;
    border-bottom: 1px solid {THEME['border']};
    padding: 10px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QSlider::groove:horizontal {{
    height: 5px; background: #2a2a50; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0;
    background: {THEME['accent']}; border-radius: 8px;
}}
QSlider::sub-page:horizontal {{ background: {THEME['accent']}; border-radius: 3px; }}
QComboBox {{
    background: {THEME['card']};
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    border-radius: 7px;
    padding: 8px 14px;
    font-size: 14px;
}}
QComboBox:focus {{ border-color: {THEME['accent']}; }}
QComboBox QAbstractItemView {{
    background: {THEME['card']};
    color: {THEME['text']};
    border: 1px solid {THEME['border']};
    selection-background-color: #28285a;
}}
QScrollBar:vertical {{
    background: #0e0e22; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #2a2a55; border-radius: 4px; min-height: 24px;
}}
QCheckBox {{ color: {THEME['text']}; spacing: 10px; font-size: 14px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 4px;
    border: 1px solid {THEME['border']}; background: {THEME['card']};
}}
QCheckBox::indicator:checked {{ background: {THEME['accent']}; border-color: {THEME['accent']}; }}
QMessageBox {{ background: {THEME['panel']}; color: {THEME['text']}; }}
"""

# =============================================================================
# APP ICON
# =============================================================================
def make_app_icon() -> QPixmap:
    size = 64
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(THEME['panel']))
    p.setPen(QPen(QColor(THEME['accent']), 2))
    p.drawEllipse(2, 2, size - 4, size - 4)
    pen = QPen(QColor(THEME['accent']), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    from PyQt5.QtGui import QPainterPath
    path = QPainterPath()
    cx, cy = size // 2, size // 2
    pts = [(8,cy),(18,cy),(24,cy-16),(30,cy+14),(36,cy-8),(40,cy),(56,cy)]
    path.moveTo(*pts[0])
    for x, y in pts[1:]: path.lineTo(x, y)
    p.drawPath(path)
    p.setBrush(QColor(THEME['danger'])); p.setPen(Qt.NoPen)
    p.drawEllipse(28, cy - 18, 6, 6)
    p.end()
    return pix

def styled_panel():
    f = QFrame(); f.setObjectName("panel"); return f

def section_label(text, color=None):
    l = QLabel(text)
    c = color or THEME['subtext']
    l.setStyleSheet(f"color:{c}; font-size:12px; font-weight:700; letter-spacing:1px;")
    return l

def heading(text, size=17, color=None):
    l = QLabel(text)
    c = color or THEME['text']
    l.setStyleSheet(f"color:{c}; font-size:{size}px; font-weight:700;")
    return l

def hsep():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{THEME['border']};"); return f

def metric_card(title, value, subtitle="", color=None):
    """Returns a styled card widget showing a single metric."""
    card = styled_panel()
    cv = QVBoxLayout(card)
    cv.setContentsMargins(16, 14, 16, 14); cv.setSpacing(4)
    t = QLabel(title)
    t.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px; font-weight:700; letter-spacing:1px;")
    v = QLabel(str(value))
    c = color or THEME['accent']
    v.setStyleSheet(f"color:{c}; font-size:26px; font-weight:800;")
    cv.addWidget(t); cv.addWidget(v)
    if subtitle:
        s = QLabel(subtitle)
        s.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        cv.addWidget(s)
    return card, v  # return widget + value label so caller can update it

# =============================================================================
# SESSION METRICS TRACKER
# =============================================================================
class SessionMetrics:
    """
    Collects all measurable data during a session.
    Call .to_dict() / .save() when session ends.

    Metrics tracked
    ---------------
    Latency          : per-inference round-trip ms  → mean, p50, p95, p99, max
    Throughput       : inferences per minute, frames per minute
    Tracking quality : % frames with full body, % left/right hand separately
    Buffer           : fill-time per window, miss rate
    Errors           : timeout count, HTTP error count
    Predictions      : top-1 confidence rolling stats, BFRB rate
    Memory           : process RAM (RSS) sampled during session
    """

    def __init__(self, session_id: int, user_id: int):
        self.session_id     = session_id
        self.user_id        = user_id
        self.start_ts       = time.time()
        self.end_ts         = None

        # Latency (ms)
        self.latencies: list[float] = []

        # Frame tracking counters
        self.frames_total        = 0
        self.frames_full_body    = 0   # pose + both hands
        self.frames_pose_only    = 0   # pose but missing >= 1 hand
        self.frames_no_pose      = 0

        # Buffer
        self.buffer_fill_times: list[float] = []  # seconds to fill each window
        self._buffer_start_ts   = None

        # Inference results
        self.inference_count     = 0
        self.bfrb_detections     = 0
        self.top1_confidences: list[float] = []
        self.class_counts: dict  = {}   # {class_name: count as top-1}

        # Errors
        self.timeout_count       = 0
        self.http_error_count    = 0

        # Process memory (RSS)
        self.ram_samples_mb: list[float] = []
        self._ram_sample_interval_s = 1.0
        self._last_ram_sample_ts = 0.0
        self._proc = None
        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
            except Exception:
                self._proc = None

    def _sample_ram(self):
        """Sample process RSS in MB at a fixed interval."""
        if self._proc is None:
            return
        now = time.time()
        if now - self._last_ram_sample_ts < self._ram_sample_interval_s:
            return
        self._last_ram_sample_ts = now
        try:
            rss_bytes = self._proc.memory_info().rss
            self.ram_samples_mb.append(rss_bytes / (1024 * 1024))
        except Exception:
            pass

    # ── Called by CameraThread ────────────────────────────────────────────
    def record_frame(self, has_body: bool, has_lhand: bool, has_rhand: bool):
        self._sample_ram()
        self.frames_total += 1
        full = has_body and has_lhand and has_rhand
        if full:
            self.frames_full_body += 1
            if self._buffer_start_ts is None:
                self._buffer_start_ts = time.time()
        elif has_body:
            self.frames_pose_only += 1
            self._buffer_start_ts = None  # reset, window broken
        else:
            self.frames_no_pose += 1
            self._buffer_start_ts = None

    def record_buffer_filled(self):
        """Called when BUFFER_FRAMES of good frames have been collected."""
        if self._buffer_start_ts is not None:
            elapsed = time.time() - self._buffer_start_ts
            self.buffer_fill_times.append(elapsed)
        self._buffer_start_ts = time.time()  # restart for next window

    # ── Called by InferenceThread ─────────────────────────────────────────
    def record_inference(self, latency_ms: float, top5: list, is_bfrb: bool,
                         timed_out=False, http_error=False):
        if timed_out:
            self.timeout_count += 1; return
        if http_error:
            self.http_error_count += 1; return

        self.inference_count += 1
        self.latencies.append(latency_ms)

        if top5:
            top_name, top_conf = top5[0]
            self.top1_confidences.append(top_conf)
            self.class_counts[top_name] = self.class_counts.get(top_name, 0) + 1
            if is_bfrb:
                self.bfrb_detections += 1

    # ── Derived stats ─────────────────────────────────────────────────────
    def _percentile(self, data: list, pct: float) -> float:
        if not data: return 0.0
        s = sorted(data)
        idx = (len(s) - 1) * pct / 100
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    @property
    def duration_s(self) -> float:
        end = self.end_ts or time.time()
        return end - self.start_ts

    @property
    def tracking_quality_pct(self) -> float:
        if self.frames_total == 0: return 0.0
        return 100.0 * self.frames_full_body / self.frames_total

    @property
    def latency_mean(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def latency_p50(self) -> float:
        return self._percentile(self.latencies, 50)

    @property
    def latency_p95(self) -> float:
        return self._percentile(self.latencies, 95)

    @property
    def latency_p99(self) -> float:
        return self._percentile(self.latencies, 99)

    @property
    def latency_max(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def latency_std(self) -> float:
        return float(np.std(self.latencies)) if len(self.latencies) > 1 else 0.0

    @property
    def inferences_per_minute(self) -> float:
        dur = self.duration_s
        return 60.0 * self.inference_count / dur if dur > 0 else 0.0

    @property
    def frames_per_minute(self) -> float:
        dur = self.duration_s
        return 60.0 * self.frames_total / dur if dur > 0 else 0.0

    @property
    def bfrb_rate_per_minute(self) -> float:
        dur = self.duration_s
        return 60.0 * self.bfrb_detections / dur if dur > 0 else 0.0

    @property
    def error_rate_pct(self) -> float:
        total = self.inference_count + self.timeout_count + self.http_error_count
        if total == 0: return 0.0
        return 100.0 * (self.timeout_count + self.http_error_count) / total

    @property
    def mean_buffer_fill_s(self) -> float:
        return float(np.mean(self.buffer_fill_times)) if self.buffer_fill_times else 0.0

    @property
    def mean_top1_confidence(self) -> float:
        return float(np.mean(self.top1_confidences)) if self.top1_confidences else 0.0

    @property
    def mean_ram_mb(self) -> float:
        return float(np.mean(self.ram_samples_mb)) if self.ram_samples_mb else 0.0

    @property
    def peak_ram_mb(self) -> float:
        return max(self.ram_samples_mb) if self.ram_samples_mb else 0.0

    @property
    def p95_ram_mb(self) -> float:
        return self._percentile(self.ram_samples_mb, 95)

    # ── Serialise ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "meta": {
                "session_id":    self.session_id,
                "user_id":       self.user_id,
                "start_time":    datetime.fromtimestamp(self.start_ts).isoformat(),
                "end_time":      datetime.fromtimestamp(self.end_ts or time.time()).isoformat(),
                "duration_s":    round(self.duration_s, 2),
            },
            "latency_ms": {
                "mean":  round(self.latency_mean, 2),
                "p50":   round(self.latency_p50, 2),
                "p95":   round(self.latency_p95, 2),
                "p99":   round(self.latency_p99, 2),
                "max":   round(self.latency_max, 2),
                "std":   round(self.latency_std, 2),
                "raw":   [round(x, 2) for x in self.latencies],
            },
            "throughput": {
                "inference_count":        self.inference_count,
                "inferences_per_minute":  round(self.inferences_per_minute, 2),
                "frames_total":           self.frames_total,
                "frames_per_minute":      round(self.frames_per_minute, 2),
            },
            "tracking_quality": {
                "frames_full_body":       self.frames_full_body,
                "frames_pose_only":       self.frames_pose_only,
                "frames_no_pose":         self.frames_no_pose,
                "tracking_quality_pct":   round(self.tracking_quality_pct, 2),
            },
            "buffer": {
                "windows_completed":      len(self.buffer_fill_times),
                "mean_fill_time_s":       round(self.mean_buffer_fill_s, 2),
                "fill_times_s":           [round(x, 3) for x in self.buffer_fill_times],
            },
            "predictions": {
                "bfrb_detections":        self.bfrb_detections,
                "bfrb_rate_per_minute":   round(self.bfrb_rate_per_minute, 2),
                "mean_top1_confidence":   round(self.mean_top1_confidence, 4),
                "class_counts":           self.class_counts,
            },
            "reliability": {
                "timeout_count":    self.timeout_count,
                "http_error_count": self.http_error_count,
                "error_rate_pct":   round(self.error_rate_pct, 2),
            },
            "memory": {
                "sample_count":  len(self.ram_samples_mb),
                "mean_ram_mb":   round(self.mean_ram_mb, 2),
                "p95_ram_mb":    round(self.p95_ram_mb, 2),
                "peak_ram_mb":   round(self.peak_ram_mb, 2),
                "samples_mb":    [round(x, 2) for x in self.ram_samples_mb],
            },
        }

    def save(self):
        """Write JSON + flat CSV row to ./metrics/ folder."""
        self.end_ts = time.time()
        d = self.to_dict()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = f"session_{self.session_id}_{ts}"

        # JSON (full detail)
        json_path = METRICS_DIR / f"{slug}.json"
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2)

        # CSV (flat summary row — easy to paste into Excel / LaTeX table)
        csv_path = METRICS_DIR / "all_sessions_metrics.csv"
        flat = {
            "session_id":             self.session_id,
            "user_id":                self.user_id,
            "start_time":             d["meta"]["start_time"],
            "duration_s":             d["meta"]["duration_s"],
            "latency_mean_ms":        d["latency_ms"]["mean"],
            "latency_p50_ms":         d["latency_ms"]["p50"],
            "latency_p95_ms":         d["latency_ms"]["p95"],
            "latency_p99_ms":         d["latency_ms"]["p99"],
            "latency_max_ms":         d["latency_ms"]["max"],
            "latency_std_ms":         d["latency_ms"]["std"],
            "inference_count":        d["throughput"]["inference_count"],
            "inferences_per_min":     d["throughput"]["inferences_per_minute"],
            "frames_total":           d["throughput"]["frames_total"],
            "frames_per_min":         d["throughput"]["frames_per_minute"],
            "tracking_quality_pct":   d["tracking_quality"]["tracking_quality_pct"],
            "frames_full_body":       d["tracking_quality"]["frames_full_body"],
            "frames_pose_only":       d["tracking_quality"]["frames_pose_only"],
            "frames_no_pose":         d["tracking_quality"]["frames_no_pose"],
            "windows_completed":      d["buffer"]["windows_completed"],
            "mean_buffer_fill_s":     d["buffer"]["mean_fill_time_s"],
            "bfrb_detections":        d["predictions"]["bfrb_detections"],
            "bfrb_rate_per_min":      d["predictions"]["bfrb_rate_per_minute"],
            "mean_top1_confidence":   d["predictions"]["mean_top1_confidence"],
            "timeout_count":          d["reliability"]["timeout_count"],
            "http_error_count":       d["reliability"]["http_error_count"],
            "error_rate_pct":         d["reliability"]["error_rate_pct"],
            "ram_sample_count":       d["memory"]["sample_count"],
            "ram_mean_mb":            d["memory"]["mean_ram_mb"],
            "ram_p95_mb":             d["memory"]["p95_ram_mb"],
            "ram_peak_mb":            d["memory"]["peak_ram_mb"],
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flat.keys())
            if write_header: w.writeheader()
            w.writerow(flat)

        print(f"[Metrics] Saved → {json_path}")
        print(f"[Metrics] Appended row → {csv_path}")
        return d

# =============================================================================
# MEDIAPIPE + FEATURE EXTRACTION  (unchanged)
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

    has_body  = results.pose_world_landmarks is not None and body_center is not None
    has_lhand = results.left_hand_landmarks  is not None
    has_rhand = results.right_hand_landmarks is not None
    return row, has_body, has_lhand, has_rhand


# =============================================================================
# CAMERA THREAD
# =============================================================================
class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object, bool, bool, bool)  # frame, results, has_body, has_lhand, has_rhand
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

                # Compute tracking flags here so signal carries them
                has_body  = results.pose_world_landmarks is not None
                has_lhand = results.left_hand_landmarks  is not None
                has_rhand = results.right_hand_landmarks is not None
                self.frame_ready.emit(annotated, results, has_body, has_lhand, has_rhand)
                self.msleep(16)
        cap.release()

    def stop(self):
        self._running = False; self.wait()

# =============================================================================
# INFERENCE THREAD
# =============================================================================
class InferenceThread(QThread):
    prediction_ready = pyqtSignal(list, float, bool)  # top5, latency_ms, is_bfrb
    inference_error  = pyqtSignal(str, bool)           # message, is_timeout

    def __init__(self, sensitivity=0.6, metrics: SessionMetrics = None):
        super().__init__()
        self.sensitivity        = sensitivity
        self.metrics            = metrics
        self._buffer            = collections.deque(maxlen=BUFFER_FRAMES)
        self._lock              = threading.Lock()
        self._running           = False
        self._trigger           = threading.Event()

    def push_frame(self, row_dict):
        with self._lock:
            self._buffer.append(row_dict)
            if len(self._buffer) >= BUFFER_FRAMES:
                if self.metrics:
                    self.metrics.record_buffer_filled()
                self._trigger.set()

    def run(self):
        self._running = True
        while self._running:
            self._trigger.wait()
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
                    "", # place the public ip here
                    json=payload, timeout=10)
                latency_ms = (time.time() - t0) * 1000
                rj = resp.json()
                if "predictions" not in rj: continue
                top5 = [(item["class"], float(item["probability"]))
                        for item in rj["predictions"]]
                top_name = top5[0][0] if top5 else ""
                top_conf = top5[0][1] if top5 else 0.0
                is_bfrb  = top_name in BFRB_CLASSES and top_conf >= self.sensitivity

                if self.metrics:
                    self.metrics.record_inference(latency_ms, top5, is_bfrb)

                self.prediction_ready.emit(top5, latency_ms, is_bfrb)
                with self._lock:
                    self._buffer.clear()

            except requests.exceptions.Timeout:
                if self.metrics: self.metrics.record_inference(0, [], False, timed_out=True)
                self.inference_error.emit("Request timed out", True)
            except requests.exceptions.RequestException as e:
                if self.metrics: self.metrics.record_inference(0, [], False, http_error=True)
                self.inference_error.emit(str(e), False)
            except Exception as e:
                print(f"[InferenceThread] {e}")

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
        label_w = 190; bar_x = label_w + 10
        bar_w = W - bar_x - 64; bar_h = 10; bar_y = (H - bar_h) // 2
        p.setPen(QColor(THEME['text'])); p.setFont(QFont("Segoe UI", 11))
        p.drawText(QRect(0, 0, label_w, H), Qt.AlignVCenter | Qt.AlignLeft, self.label_text)
        p.setBrush(QColor("#1e1e40")); p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 4, 4)
        fill_w = max(0, int(bar_w * self.value))
        if fill_w > 0:
            color = QColor(THEME['bfrb_bar']) if self.is_bfrb else QColor(THEME['norm_bar'])
            p.setBrush(color); p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 4, 4)
        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 11))
        p.drawText(QRect(bar_x + bar_w + 8, 0, 56, H),
                   Qt.AlignVCenter | Qt.AlignLeft, f"{self.value*100:.1f}%")

# =============================================================================
# LATENCY SPARKLINE WIDGET
# =============================================================================
class LatencySparkline(QWidget):
    """Rolling line chart of the last N latency measurements."""
    def __init__(self, maxlen=40, parent=None):
        super().__init__(parent)
        self._data = collections.deque(maxlen=maxlen)
        self.setMinimumHeight(60)

    def add(self, value: float):
        self._data.append(value); self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pad = 6
        p.fillRect(0, 0, W, H, QColor(THEME['card']))

        if len(self._data) < 2:
            p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRect(0, 0, W, H), Qt.AlignCenter, "No data yet")
            return

        data = list(self._data)
        mn, mx = min(data), max(data)
        if mx == mn: mx = mn + 1

        # P95 guide line
        p95 = sorted(data)[int(len(data) * 0.95)]
        y95 = pad + (H - 2*pad) * (1 - (p95 - mn)/(mx - mn))
        pen95 = QPen(QColor(THEME['warning'])); pen95.setStyle(Qt.DashLine); pen95.setWidth(1)
        p.setPen(pen95)
        p.drawLine(pad, int(y95), W - pad, int(y95))
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(THEME['warning']))
        p.drawText(W - 42, int(y95) - 2, f"p95")

        # Line
        pen = QPen(QColor(THEME['accent']), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        pts = []
        for i, v in enumerate(data):
            x = pad + i * (W - 2*pad) / (len(data) - 1)
            y = pad + (H - 2*pad) * (1 - (v - mn)/(mx - mn))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            p.drawLine(int(pts[i][0]), int(pts[i][1]), int(pts[i+1][0]), int(pts[i+1][1]))

        # Axis labels
        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRect(pad, H - 14, 60, 14), Qt.AlignLeft, f"min {mn:.0f}ms")
        p.drawText(QRect(W//2 - 30, H - 14, 60, 14), Qt.AlignCenter, f"mean {sum(data)/len(data):.0f}ms")
        p.drawText(QRect(W - 70, H - 14, 68, 14), Qt.AlignRight, f"max {mx:.0f}ms")

# =============================================================================
# MINI CHART WIDGET
# =============================================================================
class MiniBarChart(QWidget):
    def __init__(self, data: dict, title="", color=THEME['accent'], rotate_labels=False,
                 palette=None, x_axis_label="Categories", y_axis_label="Count", parent=None):
        super().__init__(parent)
        self.data          = data
        self.title         = title
        self.color         = color
        self.rotate_labels = rotate_labels
        self.palette       = palette or []
        self.x_axis_label  = x_axis_label
        self.y_axis_label  = y_axis_label
        self.setMinimumHeight(180)

    @staticmethod
    def _fmt_tick(v: float) -> str:
        if v >= 1000:
            return f"{v/1000:.1f}k"
        if abs(v - round(v)) < 0.05:
            return f"{int(round(v))}"
        return f"{v:.1f}"

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pad_l, pad_r, pad_t = 44, 14, 26
        pad_b = 70 if self.rotate_labels else 62
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_t - pad_b
        if chart_w < 10 or chart_h < 10:
            return

        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRect(pad_l, 4, W - pad_l, 18), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        items = list(self.data.items())
        if not items:
            p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 10))
            p.drawText(QRect(pad_l, pad_t, chart_w, chart_h), Qt.AlignCenter, "No data")
            return

        mx = max(v for _, v in items)
        mx = max(mx, 1.0)
        y_steps = 4
        p.setPen(QColor(THEME['border']))
        for i in range(y_steps + 1):
            y = pad_t + int(chart_h * i / y_steps)
            p.drawLine(pad_l, y, pad_l + chart_w, y)
            tick_val = mx * (1 - (i / y_steps))
            p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRect(4, y - 7, pad_l - 8, 14),
                       Qt.AlignRight | Qt.AlignVCenter, self._fmt_tick(tick_val))
            p.setPen(QColor(THEME['border']))

        p.setPen(QColor(THEME['border']))
        p.drawLine(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h)
        p.drawLine(pad_l, pad_t, pad_l, pad_t + chart_h)

        n = len(items)
        slot_w = chart_w / max(n, 1)
        bar_w = min(36, max(6, int(slot_w * 0.62)))
        label_step = max(1, int(math.ceil(n / 8)))

        for i, (label, val) in enumerate(items):
            x = pad_l + int(i * slot_w + (slot_w - bar_w) / 2)
            bar_h = int(chart_h * (val / mx))
            y = pad_t + chart_h - bar_h
            color = QColor(self.palette[i % len(self.palette)]) if self.palette else QColor(self.color)
            p.setBrush(color); p.setPen(Qt.NoPen)
            p.drawRoundedRect(x, y, bar_w, bar_h, 4, 4)

            if i % label_step == 0:
                p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
                if self.rotate_labels:
                    txt = label
                    p.save()
                    label_rect = QRect(x - 8, H - pad_b + 4, bar_w + 24, pad_b - 14)
                    p.translate(label_rect.left() + 2, label_rect.top() + 2)
                    p.rotate(-35)
                    p.drawText(QRect(0, 0, max(label_rect.width(), 1), label_rect.height()),
                               Qt.AlignLeft | Qt.AlignTop, txt)
                    p.restore()
                else:
                    # Keep labels horizontal and show full text inside each bar slot.
                    slot_x = int(pad_l + i * slot_w)
                    label_rect = QRect(slot_x + 1, H - pad_b + 4, max(int(slot_w) - 2, 1), pad_b - 16)
                    p.drawText(label_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, label)

        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRect(pad_l, H - 16, chart_w, 14), Qt.AlignCenter, self.x_axis_label)
        p.save()
        p.translate(12, pad_t + chart_h / 2)
        p.rotate(-90)
        p.drawText(QRect(-int(chart_h / 2), -10, chart_h, 14), Qt.AlignCenter, self.y_axis_label)
        p.restore()


class LineChart(QWidget):
    def __init__(self, data: dict, title="", color=THEME['accent'],
                 x_axis_label="Time", y_axis_label="Events", parent=None):
        super().__init__(parent)
        self.data = data
        self.title = title
        self.color = color
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.setMinimumHeight(200)

    @staticmethod
    def _fmt_tick(v: float) -> str:
        if abs(v - round(v)) < 0.05:
            return f"{int(round(v))}"
        return f"{v:.1f}"

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 50, 16, 28, 56
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_t - pad_b
        if chart_w < 10 or chart_h < 10:
            return

        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRect(pad_l, 4, W - pad_l, 18), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        items = list(self.data.items())
        if not items:
            p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 10))
            p.drawText(QRect(pad_l, pad_t, chart_w, chart_h), Qt.AlignCenter, "No data")
            return

        values = [v for _, v in items]
        mx, mn = max(values), min(values)
        if mx == mn:
            mn = 0
            mx = mx + 1

        y_steps = 4
        p.setPen(QColor(THEME['border']))
        for i in range(y_steps + 1):
            y = pad_t + int(chart_h * i / y_steps)
            p.drawLine(pad_l, y, pad_l + chart_w, y)
            tick_val = mx - (mx - mn) * (i / y_steps)
            p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRect(4, y - 7, pad_l - 8, 14),
                       Qt.AlignRight | Qt.AlignVCenter, self._fmt_tick(tick_val))
            p.setPen(QColor(THEME['border']))

        p.drawLine(pad_l, pad_t + chart_h, pad_l + chart_w, pad_t + chart_h)
        p.drawLine(pad_l, pad_t, pad_l, pad_t + chart_h)

        n = len(items)
        step_x = chart_w / max(n - 1, 1)
        points = []
        for i, (_, val) in enumerate(items):
            x = pad_l + i * step_x if n > 1 else pad_l + chart_w / 2
            y = pad_t + chart_h - int((val - mn) / (mx - mn) * chart_h)
            points.append((x, y))

        pen = QPen(QColor(self.color), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        for i in range(len(points) - 1):
            p.drawLine(int(points[i][0]), int(points[i][1]), int(points[i + 1][0]), int(points[i + 1][1]))
        for x, y in points:
            p.setBrush(QColor(self.color)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, y), 4, 4)

        label_step = max(1, int(math.ceil(n / 8)))
        p.setPen(QColor(THEME['subtext'])); p.setFont(QFont("Segoe UI", 8))
        for i, (label, _) in enumerate(items):
            if i % label_step != 0:
                continue
            x = pad_l + i * step_x if n > 1 else pad_l + chart_w / 2
            txt = label if len(label) <= 10 else (label[:9] + "…")
            p.drawText(QRect(int(x) - 26, pad_t + chart_h + 6, 52, 14),
                       Qt.AlignHCenter | Qt.AlignTop, txt)

        p.drawText(QRect(pad_l, H - 16, chart_w, 14), Qt.AlignCenter, self.x_axis_label)
        p.save()
        p.translate(12, pad_t + chart_h / 2)
        p.rotate(-90)
        p.drawText(QRect(-int(chart_h / 2), -10, chart_h, 14), Qt.AlignCenter, self.y_axis_label)
        p.restore()

# =============================================================================
# PERFORMANCE TAB  (for report)
# =============================================================================
class PerformanceTab(QWidget):
    """
    Aggregates metrics across all saved sessions and renders them
    in a table + charts.  Ideal for screenshotting into your report.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20); outer.setSpacing(14)

        hdr = QHBoxLayout()
        hdr.addWidget(heading("System Performance Metrics", 18))
        hdr.addStretch()
        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        export_btn = QPushButton("⬇  Export Report CSV")
        export_btn.setObjectName("primary")
        export_btn.clicked.connect(self._export)
        hdr.addWidget(refresh_btn); hdr.addWidget(export_btn)
        outer.addLayout(hdr)
        outer.addWidget(section_label(
            "Auto-saved after every session"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget(); cv = QVBoxLayout(content)
        cv.setSpacing(16); cv.setContentsMargins(0, 0, 8, 0)

        # ── Summary cards row ─────────────────────────────────────────────
        self.cards_row = QHBoxLayout(); self.cards_row.setSpacing(10)
        cv.addLayout(self.cards_row)

        # ── Latency table ─────────────────────────────────────────────────
        cv.addWidget(section_label("LATENCY BREAKDOWN  (ms)  —  per session"))
        self.latency_table = self._make_table(
            ["Session", "Mean", "P50", "P95", "P99", "Max", "Std Dev"])
        cv.addWidget(self.latency_table)

        # ── Throughput table ──────────────────────────────────────────────
        cv.addWidget(section_label("THROUGHPUT  &  TRACKING  —  per session"))
        self.thru_table = self._make_table(
            ["Session", "Duration", "Frames/min", "Inf/min",
             "Tracking %", "Full Body", "Pose Only", "No Pose"])
        cv.addWidget(self.thru_table)

        # ── Reliability table ─────────────────────────────────────────────
        cv.addWidget(section_label("RELIABILITY  &  PREDICTION QUALITY"))
        self.rel_table = self._make_table(
            ["Session", "Inferences", "BFRB Events", "BFRB/min",
             "Avg Confidence", "Timeouts", "HTTP Errors", "Error Rate %"])
        cv.addWidget(self.rel_table)

        # ── Charts row ────────────────────────────────────────────────────
        charts_row = QHBoxLayout(); charts_row.setSpacing(12)
        self.lat_chart_panel = styled_panel()
        # Holds two stacked latency charts; keep enough height to avoid cramping.
        self.lat_chart_panel.setMinimumHeight(520)
        self.lat_chart_layout = QVBoxLayout(self.lat_chart_panel)
        self.lat_chart_layout.setContentsMargins(12, 12, 12, 12)
        self.lat_chart_layout.setSpacing(10)

        self.track_chart_panel = styled_panel()
        self.track_chart_panel.setMinimumHeight(200)
        self.track_chart_layout = QVBoxLayout(self.track_chart_panel)
        self.track_chart_layout.setContentsMargins(12, 12, 12, 12)

        charts_row.addWidget(self.lat_chart_panel)
        charts_row.addWidget(self.track_chart_panel)
        cv.addLayout(charts_row)

        cv.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.refresh()

    def _make_table(self, headers):
        t = QTableWidget()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        t.setStyleSheet(
            t.styleSheet() + "QTableWidget { alternate-background-color: #0d0d20; }")
        t.setMinimumHeight(200)
        t.setMaximumHeight(260)
        return t

    def _load_sessions(self) -> list[dict]:
        sessions = []
        for fp in sorted(METRICS_DIR.glob("session_*.json")):
            try:
                with open(fp) as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
        sessions.sort(key=lambda s: s.get("meta", {}).get("session_id", 0))
        return sessions

    def refresh(self):
        sessions = self._load_sessions()

        # Clear cards
        while self.cards_row.count():
            item = self.cards_row.takeAt(0)
            if item.widget(): item.widget().setParent(None)

        if not sessions:
            lbl = QLabel("No sessions recorded yet. Run a session to collect metrics.")
            lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px;")
            self.cards_row.addWidget(lbl)
            for t in [self.latency_table, self.thru_table, self.rel_table]:
                t.setRowCount(0)
            return

        # ── Aggregate across all sessions ─────────────────────────────────
        all_lat   = [s["latency_ms"]["mean"]               for s in sessions if s["latency_ms"]["mean"]]
        all_track = [s["tracking_quality"]["tracking_quality_pct"] for s in sessions]
        all_inf   = [s["throughput"]["inference_count"]    for s in sessions]
        all_err   = [s["reliability"]["error_rate_pct"]    for s in sessions]

        def _mc(title, val, sub="", color=None):
            c, v = metric_card(title, val, sub, color)
            self.cards_row.addWidget(c)

        _mc("AVG LATENCY", f"{np.mean(all_lat):.0f} ms",
            f"across {len(sessions)} session(s)", THEME['accent'])
        _mc("AVG TRACKING", f"{np.mean(all_track):.1f}%",
            "frames with full body", THEME['success'])
        _mc("TOTAL INFERENCES", str(sum(all_inf)),
            "across all sessions", THEME['accent2'])
        _mc("AVG ERROR RATE", f"{np.mean(all_err):.1f}%",
            "timeouts + HTTP errors", THEME['warning'])

        # ── Latency table ─────────────────────────────────────────────────
        self.latency_table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            sid = s["meta"]["session_id"]
            lm  = s["latency_ms"]
            vals = [f"#{sid}", f"{lm['mean']:.1f}", f"{lm['p50']:.1f}",
                    f"{lm['p95']:.1f}", f"{lm['p99']:.1f}",
                    f"{lm['max']:.1f}", f"{lm['std']:.1f}"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                # Colour-code p95 > 1000ms as warning
                if j == 4 and lm['p95'] > 1000:
                    item.setForeground(QColor(THEME['danger']))
                self.latency_table.setItem(i, j, item)

        # ── Throughput table ──────────────────────────────────────────────
        self.thru_table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            sid = s["meta"]["session_id"]
            t   = s["throughput"]; tq = s["tracking_quality"]
            dur = s["meta"]["duration_s"]
            dur_str = f"{int(dur//60)}m {int(dur%60)}s"
            vals = [f"#{sid}", dur_str,
                    f"{t['frames_per_minute']:.1f}", f"{t['inferences_per_minute']:.2f}",
                    f"{tq['tracking_quality_pct']:.1f}%",
                    str(tq['frames_full_body']), str(tq['frames_pose_only']),
                    str(tq['frames_no_pose'])]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                if j == 4:
                    pct = tq['tracking_quality_pct']
                    item.setForeground(QColor(
                        THEME['success'] if pct >= 80 else
                        THEME['warning'] if pct >= 50 else THEME['danger']))
                self.thru_table.setItem(i, j, item)

        # ── Reliability table ─────────────────────────────────────────────
        self.rel_table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            sid = s["meta"]["session_id"]
            th  = s["throughput"]; pr = s["predictions"]; re = s["reliability"]
            vals = [f"#{sid}", str(th['inference_count']),
                    str(pr['bfrb_detections']), f"{pr['bfrb_rate_per_minute']:.2f}",
                    f"{pr['mean_top1_confidence']:.3f}",
                    str(re['timeout_count']), str(re['http_error_count']),
                    f"{re['error_rate_pct']:.1f}%"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                if j == 7 and re['error_rate_pct'] > 5:
                    item.setForeground(QColor(THEME['danger']))
                self.rel_table.setItem(i, j, item)

        # ── Latency bar chart (mean per session) ──────────────────────────
        self._clear_layout(self.lat_chart_layout)
        lat_data = {f"#{s['meta']['session_id']}": s["latency_ms"]["mean"]
                    for s in sessions}
        p95_data = {f"#{s['meta']['session_id']}": s["latency_ms"]["p95"]
                    for s in sessions}
        self.lat_chart_layout.addWidget(
            section_label("MEAN LATENCY PER SESSION (ms)"))
        mean_chart = MiniBarChart(lat_data, "Mean latency (ms)", THEME['accent'],
                                  x_axis_label="Session ID", y_axis_label="Latency (ms)")
        mean_chart.setMinimumHeight(220)
        self.lat_chart_layout.addWidget(mean_chart)
        p95_chart = MiniBarChart(p95_data, "P95 latency (ms)", THEME['warning'],
                                 x_axis_label="Session ID", y_axis_label="Latency (ms)")
        p95_chart.setMinimumHeight(220)
        self.lat_chart_layout.addWidget(p95_chart)

        # ── Tracking quality bar chart ────────────────────────────────────
        self._clear_layout(self.track_chart_layout)
        track_data = {f"#{s['meta']['session_id']}": s["tracking_quality"]["tracking_quality_pct"]
                      for s in sessions}
        self.track_chart_layout.addWidget(
            section_label("TRACKING QUALITY % PER SESSION"))
        self.track_chart_layout.addWidget(
            MiniBarChart(track_data, "Full-body tracking (%)", THEME['success'],
                         x_axis_label="Session ID", y_axis_label="Tracking (%)"))

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)

    def _export(self):
        csv_path = METRICS_DIR / "all_sessions_metrics.csv"
        if not csv_path.exists():
            QMessageBox.warning(self, "No Data", "No metrics CSV found. Run some sessions first.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics Report", "bfrb_metrics_report.csv", "CSV Files (*.csv)")
        if not dest: return
        import shutil
        shutil.copy(csv_path, dest)
        QMessageBox.information(self, "Exported", f"Report saved to:\n{dest}")

# =============================================================================
# SHARED DASHBOARD WIDGET
# =============================================================================
class DashboardWidget(QWidget):
    session_started = pyqtSignal(int)
    session_ended   = pyqtSignal(int, int)

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
        self._last_seen_event_id = None
        self._user_events_initialized = False
        self._last_user_popup_time = 0.0
        self._user_popup_cooldown_s = 5.0
        self._active_alert_popups = []
        self._last_user_analytics_refresh = 0.0
        self._user_alert_banner_timer = None
        self._user_ack_dialog_open = False
        self.prefs            = db.get_preferences(current_user["user_id"])
        self.session_metrics  = None   # SessionMetrics instance

        self._build()
        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._update_time_ago)
        self._tick_timer.start(10000)

        if not self.is_admin:
            self._poll_timer = QTimer()
            self._poll_timer.timeout.connect(self._poll_user_stats)
            self._poll_timer.start(1000)

    def _build(self):
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(14)

        # ── ADMIN MODE: Video + Controls Layout ──────────────────────────
        if self.is_admin:
            # Left: video
            left = styled_panel(); lv = QVBoxLayout(left)
            lv.setContentsMargins(10,10,10,10); lv.setSpacing(8)

            self.video_label = QLabel()
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.video_label.setMinimumSize(480, 360)
            self.video_label.setStyleSheet(
                "background:#060612; border-radius:8px; color:#444466; font-size:15px;")
            self.video_label.setText("No active session")
            lv.addWidget(self.video_label, stretch=1)

            stat_row = QHBoxLayout()
            self.fps_lbl     = QLabel("FPS: —")
            self.latency_lbl = QLabel("")
            self.track_lbl   = QLabel("● Idle")
            for lbl in [self.fps_lbl, self.latency_lbl, self.track_lbl]:
                lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px;")
            stat_row.addWidget(self.fps_lbl)
            stat_row.addStretch()
            stat_row.addWidget(self.track_lbl)
            lv.addLayout(stat_row)

            self.sparkline = None

            root.addWidget(left, stretch=5)

            # ── Right: Admin controls + predictions ───────────────────────
            right_container = QWidget()
            right_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            right_outer = QVBoxLayout(right_container)
            right_outer.setContentsMargins(0, 0, 0, 0); right_outer.setSpacing(0)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setStyleSheet(
                "QScrollArea { border: none; background: transparent; }"
                f"QScrollBar:vertical {{ background: #0e0e22; width: 6px; border-radius: 3px; }}"
                f"QScrollBar::handle:vertical {{ background: #2a2a55; border-radius: 3px; min-height: 20px; }}"
            )

            scroll_contents = QWidget()
            scroll_contents.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            right = QVBoxLayout(scroll_contents)
            right.setSpacing(12); right.setContentsMargins(0, 0, 6, 0)

            # ── Session control (admin only) ──────────────────────────────
            ctrl = styled_panel(); cv = QVBoxLayout(ctrl)
            cv.setContentsMargins(16,14,16,14); cv.setSpacing(10)
            cv.addWidget(heading("Session Control", 15, THEME['accent']))
            cv.addWidget(section_label("SELECT USER"))
            self.user_combo = QComboBox()
            self._refresh_user_combo()
            cv.addWidget(self.user_combo)
            cv.addWidget(hsep())
            btn_row = QHBoxLayout(); btn_row.setSpacing(8)
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
            self.session_status.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px;")
            cv.addWidget(self.session_status)
            right.addWidget(ctrl)

            # ── Live Stats panel ──────────────────────────────────────────
            stats = styled_panel(); sv = QVBoxLayout(stats)
            sv.setContentsMargins(16,14,16,14); sv.setSpacing(10)
            sv.addWidget(heading("Live Stats", 15, THEME['accent']))

            grid = QGridLayout(); grid.setSpacing(10)
            grid.addWidget(section_label("BFRB EVENTS"), 0, 0)
            grid.addWidget(section_label("LAST DETECTION"), 0, 1)

            self.bfrb_lbl = QLabel("0")
            self.bfrb_lbl.setStyleSheet(
                f"font-size:42px; font-weight:800; color:{THEME['danger']};")
            self.last_det_lbl = QLabel("—")
            self.last_det_lbl.setStyleSheet(
                f"font-size:14px; color:{THEME['text']}; font-weight:700;")
            self.last_det_lbl.setWordWrap(True)
            self.time_ago_lbl = QLabel("")
            self.time_ago_lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px;")

            grid.addWidget(self.bfrb_lbl, 1, 0)
            det_col = QVBoxLayout()
            det_col.addWidget(self.last_det_lbl)
            det_col.addWidget(self.time_ago_lbl)
            det_col.setSpacing(3)
            det_w = QWidget(); det_w.setLayout(det_col)
            grid.addWidget(det_w, 1, 1)
            sv.addLayout(grid)

            self.badge = QLabel("Waiting for inference…")
            self.badge.setFixedHeight(32)
            self.badge.setAlignment(Qt.AlignCenter)
            self.badge.setStyleSheet(
                f"background:#1a1a38; color:{THEME['subtext']}; border:1px solid {THEME['border']};"
                "border-radius:6px; font-size:13px; font-weight:600; padding:2px 12px;")
            sv.addWidget(self.badge)

            self.top_label = QLabel("—")
            self.top_label.setStyleSheet(
                f"font-size:22px; font-weight:800; color:#fff;"
                f"background:{THEME['card']}; border-radius:10px; padding:12px 16px;")
            self.top_label.setWordWrap(True)
            self.top_label.setMinimumHeight(52)
            sv.addWidget(self.top_label)

            sv.addWidget(hsep())
            sv.addWidget(section_label("TOP 5 PREDICTIONS"))
            self.conf_bars = []
            for _ in range(5):
                bar = ConfidenceBar()
                sv.addWidget(bar)
                self.conf_bars.append(bar)

            right.addWidget(stats)
            right.addStretch()

            scroll.setWidget(scroll_contents)
            right_outer.addWidget(scroll)
            root.addWidget(right_container, stretch=3)

        # ── USER MODE: Dashboard Statistics Layout ───────────────────────
        else:
            # Create a full-width dashboard for users
            dashboard = QWidget()
            dashboard.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            dash_layout = QVBoxLayout(dashboard)
            dash_layout.setContentsMargins(20, 20, 20, 20); dash_layout.setSpacing(20)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setStyleSheet(
                "QScrollArea { border: none; background: transparent; }"
                f"QScrollBar:vertical {{ background: #0e0e22; width: 8px; border-radius: 4px; }}"
                f"QScrollBar::handle:vertical {{ background: #3a3a6a; border-radius: 4px; min-height: 40px; }}"
            )

            scroll_contents = QWidget()
            scroll_contents.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            scroll_layout = QVBoxLayout(scroll_contents)
            scroll_layout.setSpacing(16); scroll_layout.setContentsMargins(0, 0, 12, 0)

            # ── Title ─────────────────────────────────────────────────────
            title = QLabel("Your BFRB Dashboard")
            title.setStyleSheet(
                f"font-size:28px; font-weight:800; color:{THEME['accent']}; letter-spacing:1px;")
            scroll_layout.addWidget(title)

            # ── Live alert banner (user mode) ─────────────────────────────
            self.user_alert_banner = QLabel("")
            self.user_alert_banner.setVisible(False)
            self.user_alert_banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.user_alert_banner.setWordWrap(True)
            self.user_alert_banner.setStyleSheet(
                f"background:#3a0f1e; color:{THEME['danger']}; border:1px solid #771530;"
                "border-radius:8px; font-size:14px; font-weight:700; padding:10px 12px;")
            scroll_layout.addWidget(self.user_alert_banner)

            # ── Main Stats Cards Row (compact) ────────────────────────────
            stats_row = QHBoxLayout(); stats_row.setSpacing(10)

            # BFRB Count Card
            bfrb_card = styled_panel(); bfrb_card_layout = QVBoxLayout(bfrb_card)
            bfrb_card_layout.setContentsMargins(16, 14, 16, 14); bfrb_card_layout.setSpacing(4)
            
            bfrb_label = QLabel("BFRB Events")
            bfrb_label.setStyleSheet(f"font-size:14px; color:{THEME['subtext']}; font-weight:600;")
            bfrb_label.setAlignment(Qt.AlignLeft)
            
            self.bfrb_lbl = QLabel("0")
            self.bfrb_lbl.setStyleSheet(
                f"font-size:44px; font-weight:900; color:{THEME['danger']}; letter-spacing:1px;")
            self.bfrb_lbl.setAlignment(Qt.AlignLeft)
            
            bfrb_card_layout.addWidget(bfrb_label)
            bfrb_card_layout.addWidget(self.bfrb_lbl)
            bfrb_card.setMinimumHeight(120)
            stats_row.addWidget(bfrb_card, 1)

            # Last Detection Card
            last_card = styled_panel(); last_card_layout = QVBoxLayout(last_card)
            last_card_layout.setContentsMargins(16, 14, 16, 14); last_card_layout.setSpacing(6)
            
            last_title = QLabel("Last Detected")
            last_title.setStyleSheet(f"font-size:14px; color:{THEME['subtext']}; font-weight:600;")
            last_title.setAlignment(Qt.AlignLeft)
            
            self.last_det_lbl = QLabel("—")
            self.last_det_lbl.setStyleSheet(
                f"font-size:20px; font-weight:800; color:{THEME['accent']};")
            self.last_det_lbl.setAlignment(Qt.AlignLeft)
            self.last_det_lbl.setWordWrap(True)
            
            self.time_ago_lbl = QLabel("")
            self.time_ago_lbl.setStyleSheet(f"font-size:13px; color:{THEME['success']}; font-weight:600;")
            self.time_ago_lbl.setAlignment(Qt.AlignLeft)
            
            last_card_layout.addWidget(last_title)
            last_card_layout.addWidget(self.last_det_lbl)
            last_card_layout.addWidget(self.time_ago_lbl)
            last_card.setMinimumHeight(120)
            stats_row.addWidget(last_card, 1)

            scroll_layout.addLayout(stats_row)

            # ── BFRB Analytics ──────────────────────────────────────────────
            analytics_card = styled_panel(); analytics_layout = QVBoxLayout(analytics_card)
            analytics_layout.setContentsMargins(20, 16, 20, 16); analytics_layout.setSpacing(12)

            analytics_layout.addWidget(section_label("YOUR BFRB ANALYTICS"))

            self.user_total_lbl = QLabel("Total detected events: 0")
            self.user_total_lbl.setStyleSheet(f"color:{THEME['text']}; font-size:16px; font-weight:700;")
            analytics_layout.addWidget(self.user_total_lbl)

            self.user_top_behavior_lbl = QLabel("Most frequent BFRB: —")
            self.user_top_behavior_lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px; font-weight:600;")
            analytics_layout.addWidget(self.user_top_behavior_lbl)

            self.behavior_breakdown_chart = MiniBarChart(
                {}, "Top BFRBs detected", THEME['bfrb_bar'], rotate_labels=False,
                palette=[THEME['bfrb_bar'], THEME['accent'], THEME['accent2'], THEME['warning'], THEME['success']],
                x_axis_label="Behavior", y_axis_label="Event count")
            self.behavior_breakdown_chart.setMinimumHeight(220)
            analytics_layout.addWidget(self.behavior_breakdown_chart)

            self.time_series_chart = LineChart(
                {}, "BFRB events over time", THEME['accent2'],
                x_axis_label="Date (MM-DD)", y_axis_label="Events")
            self.time_series_chart.setMinimumHeight(220)
            analytics_layout.addWidget(self.time_series_chart)

            scroll_layout.addWidget(analytics_card)
            scroll_layout.addStretch()

            # Add pseudo-elements for compatibility with admin mode
            self.fps_lbl = QLabel("")
            self.latency_lbl = QLabel("")
            self.track_lbl = QLabel("")
            self.sparkline = None
            self.video_label = None
            self.badge = None
            self.top_label = QLabel("")
            self.conf_bars = []

            scroll.setWidget(scroll_contents)
            dash_layout.addWidget(scroll)
            root.addWidget(dashboard)
            self._refresh_user_analytics()

    def _refresh_user_combo(self):
        if not self.is_admin: return
        self.user_combo.clear()
        users = db.get_non_admin_users()
        for u in users:
            self.user_combo.addItem(f"{u['username']} ({u['email']})", u['user_id'])
        if not users:
            self.user_combo.addItem("No users available", -1)

    # ── Session control ───────────────────────────────────────────────────
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
        self.session_metrics = SessionMetrics(self.session_id, uid)
        self.session_started.emit(self.session_id)

        self.cam_thread = CameraThread(
            show_skeleton=bool(prefs.get("skeleton_overlay", 1)))
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.error.connect(lambda e: self.session_status.setText(f"Error: {e}"))
        self.cam_thread.start()

        self.infer_thread = InferenceThread(
            sensitivity=prefs.get("sensitivity", 0.6),
            metrics=self.session_metrics)
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

        # Save metrics
        if self.session_metrics:
            self.session_metrics.save()
            self.session_metrics = None

        self.session_id = None; self.session_user_id = None
        self.video_label.clear()
        self.video_label.setStyleSheet(
            "background:#060612; border-radius:8px; color:#444466; font-size:15px;")
        self.video_label.setText("No active session")
        self.track_lbl.setText("● Idle")
        self.track_lbl.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px;")
        self.fps_lbl.setText("FPS: —"); self.latency_lbl.setText("")
        if self.is_admin:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("⏸  Pause")
            self.stop_btn.setEnabled(False)
            self.user_combo.setEnabled(True)
            self.session_status.setText("Session ended — metrics saved to ./metrics/")
            self.session_status.setStyleSheet(f"color:{THEME['subtext']}; font-size:13px;")
        self._paused = False

    # ── Frame handler ─────────────────────────────────────────────────────
    def _on_frame(self, frame, results, has_body, has_lhand, has_rhand):
        now = time.time()
        self.fps_times.append(now)
        if len(self.fps_times) > 1:
            fps = (len(self.fps_times)-1)/(self.fps_times[-1]-self.fps_times[0])
            if self.fps_lbl:
                self.fps_lbl.setText(f"FPS: {fps:.1f}")

        if self.session_metrics:
            self.session_metrics.record_frame(has_body, has_lhand, has_rhand)

        row, _, _, _ = process_holistic_results(results)
        full = has_body and has_lhand and has_rhand

        if full:
            self.missing_counter = 0
            if self.track_lbl:
                self.track_lbl.setText("● Full body — inference running")
                self.track_lbl.setStyleSheet(f"color:{THEME['success']}; font-size:11px;")
            if self.infer_thread: self.infer_thread.push_frame(row)
        else:
            self.missing_counter += 1
            if self.missing_counter >= 10:
                if self.track_lbl:
                    self.track_lbl.setText("● Please show full body")
                    self.track_lbl.setStyleSheet(f"color:{THEME['danger']}; font-size:11px;")
                if self.infer_thread: self.infer_thread._buffer.clear()
            else:
                if self.track_lbl:
                    self.track_lbl.setText(f"● Lost tracking ({self.missing_counter}/10)")
                    self.track_lbl.setStyleSheet(f"color:{THEME['warning']}; font-size:11px;")

        # Update video feed (admin mode only)
        if self.video_label:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
            self.video_label.setAlignment(Qt.AlignCenter)

    # ── Prediction handler ─────────────────────────────────────────────────
    def _on_prediction(self, top5, latency, is_bfrb):
        if not top5: return
        top_name, top_prob = top5[0]
        sensitivity = self.prefs.get("sensitivity", 0.6)
        cooldown    = self.prefs.get("alert_cooldown", 10)

        self.top_label.setText(top_name)
        if self.latency_lbl:
            self.latency_lbl.setText("")

        if is_bfrb:
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

    def _poll_user_stats(self):
        if self.is_admin: return
        uid = self.current_user["user_id"]
        sessions = db.get_sessions_for_user(uid)
        if not sessions: return
        latest = sessions[0]
        sid = latest["session_id"]
        event_count = latest.get("event_count", 0) or 0
        self.bfrb_lbl.setText(str(event_count))
        events = db.get_events_for_session(sid)
        if events:
            latest_event = events[-1]
            latest_event_id = latest_event.get("event_id")
            is_new_event = (
                self._user_events_initialized
                and latest_event_id is not None
                and latest_event_id != self._last_seen_event_id
            )
            if latest_event_id is not None:
                self._last_seen_event_id = latest_event_id
            self._user_events_initialized = True

            self.last_det_lbl.setText(latest_event["behavior_type"])
            try:
                ts = datetime.fromisoformat(latest_event["timestamp"])
                self.last_detection = ts
                delta = datetime.now() - ts
                mins = int(delta.total_seconds() // 60)
                if mins == 0:   self.time_ago_lbl.setText("just now")
                elif mins == 1: self.time_ago_lbl.setText("1 min ago")
                else:           self.time_ago_lbl.setText(f"{mins} mins ago")
            except Exception:
                pass
            if is_new_event:
                self._show_user_alert_popup(latest_event)
        end_time = latest.get("end_time")
        if end_time and self.badge:
            self.badge.setText("Session ended")
            self.badge.setStyleSheet(
                f"background:#1a1a38; color:{THEME['subtext']}; border:1px solid {THEME['border']};"
                "border-radius:6px; font-size:13px; font-weight:600; padding:2px 12px;")
        now = time.time()
        if now - self._last_user_analytics_refresh >= 5.0:
            self._last_user_analytics_refresh = now
            self._refresh_user_analytics()

    def _show_user_alert_popup(self, event: dict):
        """User-side visual alert for newly detected BFRB events."""
        if self._user_ack_dialog_open:
            return

        now = time.time()
        if now - self._last_user_popup_time < self._user_popup_cooldown_s:
            return
        self._last_user_popup_time = now

        behavior = event.get("behavior_type", "BFRB")
        confidence = event.get("confidence_score")
        if confidence is not None:
            conf_txt = f"{float(confidence) * 100:.1f}%"
        else:
            conf_txt = "N/A"

        # Always-visible in-dashboard banner for reliable visual feedback.
        if hasattr(self, "user_alert_banner") and self.user_alert_banner:
            self.user_alert_banner.setText(
                f"⚠ BFRB detected: {behavior}   (confidence: {conf_txt})")
            self.user_alert_banner.setVisible(True)

            if self._user_alert_banner_timer is None:
                self._user_alert_banner_timer = QTimer(self)
                self._user_alert_banner_timer.setSingleShot(True)
                self._user_alert_banner_timer.timeout.connect(
                    lambda: self.user_alert_banner.setVisible(False)
                )
            self._user_alert_banner_timer.start(3500)

        self._show_user_ack_dialog(behavior, conf_txt)

    def _show_user_ack_dialog(self, behavior: str, conf_txt: str):
        """Prompt user to acknowledge and interrupt the detected habit."""
        self._user_ack_dialog_open = True

        dialog = QDialog(self)
        dialog.setModal(True)
        dialog.setWindowTitle("Habit Reversal Prompt")
        dialog.setMinimumWidth(420)
        dialog.setStyleSheet(
            f"QDialog {{ background:{THEME['panel']}; color:{THEME['text']}; }}"
            f"QLabel {{ color:{THEME['text']}; font-size:14px; }}"
            f"QCheckBox {{ color:{THEME['text']}; font-size:14px; font-weight:600; }}"
            f"QPushButton {{ background:{THEME['accent']}; color:#fff; border:none;"
            "border-radius:6px; padding:8px 16px; font-weight:700; }}"
            "QPushButton:disabled { background:#3a3a60; color:#a0a0bf; }"
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("BFRB detected - pause and reset")
        title.setStyleSheet(f"font-size:16px; font-weight:800; color:{THEME['danger']};")
        layout.addWidget(title)

        msg = QLabel(
            f"Detected behavior: {behavior}\n"
            f"Confidence: {conf_txt}\n\n"
            "Take a breath, relax your hands, and choose an alternative response."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        acknowledge_cb = QCheckBox(f"I will stop {behavior.lower()} now")
        layout.addWidget(acknowledge_cb)

        yes_btn = QPushButton("Yes, I understand")
        yes_btn.setEnabled(False)
        yes_btn.clicked.connect(dialog.accept)
        acknowledge_cb.toggled.connect(yes_btn.setEnabled)
        layout.addWidget(yes_btn, alignment=Qt.AlignRight)

        dialog.finished.connect(lambda _: setattr(self, "_user_ack_dialog_open", False))
        dialog.exec_()

    def refresh_prefs(self, prefs: dict):
        self.prefs = prefs
        if self.cam_thread:
            self.cam_thread.show_skeleton = bool(prefs.get("skeleton_overlay", 1))
        self._refresh_user_analytics()

    def _refresh_user_analytics(self):
        if self.is_admin: return
        uid = self.current_user["user_id"]
        events = db.get_events_for_user(uid)
        total_events = len(events)
        self.user_total_lbl.setText(f"Total detected events: {total_events}")

        counts = {}
        daily = {}
        for e in events:
            bt = e.get("behavior_type", "Unknown")
            counts[bt] = counts.get(bt, 0) + 1
            ts = e.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    d = dt.strftime("%m-%d")
                except Exception:
                    continue
                daily[d] = daily.get(d, 0) + 1

        if counts:
            top_behavior = max(counts, key=counts.get)
            self.user_top_behavior_lbl.setText(
                f"Most frequent BFRB: {top_behavior} ({counts[top_behavior]} times)")
        else:
            self.user_top_behavior_lbl.setText("Most frequent BFRB: —")

        breakdown = dict(sorted(counts.items(), key=lambda x: -x[1])[:6])
        self.behavior_breakdown_chart.data = breakdown
        self.behavior_breakdown_chart.update()

        time_series = dict(sorted(daily.items(), key=lambda x: x[0]))
        self.time_series_chart.data = time_series
        self.time_series_chart.update()

    def cleanup(self):
        self._tick_timer.stop()
        if not self.is_admin and hasattr(self, '_poll_timer'):
            self._poll_timer.stop()
        if self.cam_thread:   self.cam_thread.stop()
        if self.infer_thread: self.infer_thread.stop()
        if self.session_id:   db.end_session(self.session_id, self.event_count)
        if self.session_metrics: self.session_metrics.save()

# =============================================================================
# SETTINGS TAB (unchanged)
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
        sensitivity    = self.sens_slider.value() / 100.0
        alert_cooldown = self.cool_slider.value()
        skeleton       = 1 if self.skel_check.isChecked() else 0
        dur = self.prefs.get("session_duration", 60)
        db.save_preferences(self.user["user_id"], sensitivity, alert_cooldown, dur, skeleton)
        self.prefs = db.get_preferences(self.user["user_id"])
        self.prefs_saved.emit(self.prefs)
        QMessageBox.information(self, "Saved", "Settings saved successfully.")

# =============================================================================
# SESSION HISTORY TAB (unchanged)
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

        hdr = QHBoxLayout()
        hdr.addWidget(heading("Session History", 18))
        hdr.addStretch()
        export_csv  = QPushButton("⬇  Export CSV")
        export_json = QPushButton("⬇  Export JSON")
        export_csv.clicked.connect(lambda: self._export("csv"))
        export_json.clicked.connect(lambda: self._export("json"))
        hdr.addWidget(export_csv); hdr.addWidget(export_json)
        outer.addLayout(hdr)
        if self.is_admin:
            outer.addWidget(section_label("Showing all users — admin view"))
        else:
            outer.addWidget(section_label("Showing your sessions only"))

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Session ID", "User" if self.is_admin else "Date",
             "Start Time", "End Time", "Duration", "Events", "Most Detected BFRB"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            self.table.styleSheet() +
            "QTableWidget { alternate-background-color: #0d0d20; }")
        self.table.verticalHeader().setVisible(False)
        outer.addWidget(self.table)
        self.refresh()

    def refresh(self):
        sessions = db.get_all_sessions() if self.is_admin else \
                   db.get_sessions_for_user(self.user["user_id"])
        self.table.setRowCount(len(sessions))
        self._sessions = sessions
        for i, s in enumerate(sessions):
            sid = s["session_id"]
            dur = s.get("duration_seconds") or 0
            dur_str = f"{dur//60}m {dur%60}s" if dur else "—"
            events = db.get_events_for_session(sid)
            behavior_counts = {}
            for e in events:
                bt = e.get("behavior_type", "Unknown")
                behavior_counts[bt] = behavior_counts.get(bt, 0) + 1
            if behavior_counts:
                most_behavior = max(behavior_counts, key=behavior_counts.get)
                most_behavior = f"{most_behavior} ({behavior_counts[most_behavior]})"
            else:
                most_behavior = "—"
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
                start_fmt, end, dur_str, str(s.get("event_count", 0)),
                most_behavior,
            ]
            for j, val in enumerate(row_data):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, item)

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
        chart = MiniBarChart(data, title=title, color=color,
                             x_axis_label="Behavior", y_axis_label="Event count")
        chart.setMinimumHeight(170)
        self.chart_layout.addWidget(chart)

    def _export(self, fmt):
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
# ADMIN USERS TAB (unchanged)
# =============================================================================
class AdminUsersTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20); outer.setSpacing(16)
        outer.addWidget(heading("User Management", 18))

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
        self.f_role = QComboBox(); self.f_role.addItems(["user", "admin"])
        add_btn = QPushButton("Add User")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._add_user)
        row2.addWidget(self.f_password); row2.addWidget(self.f_role)
        row2.addWidget(add_btn)
        fv.addLayout(row2)
        outer.addWidget(form)

        outer.addWidget(section_label("ALL ACCOUNTS"))
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Username", "Email", "Role", "Created", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(44)
        outer.addWidget(self.table)
        self.refresh()

    def refresh(self):
        users = db.get_all_users()
        self.table.setRowCount(len(users))
        for i, u in enumerate(users):
            self.table.setRowHeight(i, 44)
            vals = [str(u["user_id"]), u["username"], u["email"],
                    u["account_type"], u["created_at"][:10] if u["created_at"] else "—"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                if j == 3:
                    item.setForeground(
                        QColor(THEME['accent']) if v == "admin" else QColor(THEME['success']))
                self.table.setItem(i, j, item)
            del_btn = QPushButton("Delete"); del_btn.setObjectName("danger")
            del_btn.setFixedHeight(30)
            uid = u["user_id"]
            del_btn.clicked.connect(lambda _, uid=uid: self._delete_user(uid))
            if u["account_type"] == "admin": del_btn.setEnabled(False)
            container = QWidget(); cl = QHBoxLayout(container)
            cl.setContentsMargins(8, 6, 8, 6); cl.addWidget(del_btn)
            self.table.setCellWidget(i, 5, container)

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
# MAIN WINDOW
# =============================================================================
class AppWindow(QMainWindow):
    def __init__(self, user: dict):
        super().__init__()
        self.user     = user
        self.is_admin = user["account_type"] == "admin"
        self.setWindowTitle(
            f"BFRB Detector  —  {'Admin' if self.is_admin else user['username']}")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(APP_STYLE)
        self.setWindowIcon(QIcon(make_app_icon()))
        self._build()

    def _build(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Top bar
        topbar = QWidget(); topbar.setFixedHeight(52)
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
        logout_btn = QPushButton("Log out"); logout_btn.setFixedWidth(110)
        logout_btn.clicked.connect(self._logout)
        tb.addWidget(logo); tb.addWidget(role_badge); tb.addWidget(user_lbl)
        tb.addStretch(); tb.addWidget(logout_btn)
        root.addWidget(topbar)

        # Tabs
        self.tabs = QTabWidget(); self.tabs.setContentsMargins(12, 12, 12, 12)
        content = QWidget(); cl = QVBoxLayout(content)
        cl.setContentsMargins(12, 12, 12, 12); cl.addWidget(self.tabs)
        root.addWidget(content)

        self.dashboard = DashboardWidget(self.user, is_admin=self.is_admin)
        self.tabs.addTab(self.dashboard, "  Dashboard  ")

        self.history = HistoryTab(self.user, is_admin=self.is_admin)
        self.tabs.addTab(self.history, "  Session History  ")

        self.settings_tab = SettingsTab(self.user)
        self.settings_tab.prefs_saved.connect(self.dashboard.refresh_prefs)
        self.tabs.addTab(self.settings_tab, "  Settings  ")

        if self.is_admin:
            self.perf_tab = PerformanceTab()
            self.tabs.addTab(self.perf_tab, "  Performance  ")
            self.users_tab = AdminUsersTab()
            self.tabs.addTab(self.users_tab, "  Users  ")

        self.tabs.currentChanged.connect(self._on_tab_change)

    def _on_tab_change(self, idx):
        widget = self.tabs.widget(idx)
        if isinstance(widget, HistoryTab): widget.refresh()
        if isinstance(widget, PerformanceTab): widget.refresh()
        if self.is_admin and isinstance(widget, AdminUsersTab): widget.refresh()
        if isinstance(widget, DashboardWidget) and self.is_admin:
            self.dashboard._refresh_user_combo()

    def _logout(self):
        self.dashboard.cleanup()
        self.close()
        self._login_window = LoginWindow()
        self._login_window.show()

    def closeEvent(self, event):
        self.dashboard.cleanup(); event.accept()

# =============================================================================
# LOGIN WINDOW (unchanged)
# =============================================================================
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFRB Detector — Login")
        self.setFixedSize(460, 540)
        self.setStyleSheet(APP_STYLE)
        self.setWindowIcon(QIcon(make_app_icon()))
        self._build()

    def _build(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        card = QWidget(); card.setFixedSize(460, 540)
        card.setStyleSheet(f"background:{THEME['panel']}; border-radius:0px;")
        cl = QVBoxLayout(card); cl.setContentsMargins(50, 50, 50, 50); cl.setSpacing(0)
        logo = QLabel("BFRB"); logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"color:{THEME['accent']}; font-size:42px; font-weight:800; letter-spacing:8px;")
        cl.addWidget(logo)
        sub = QLabel("Real-Time Detection System"); sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{THEME['subtext']}; font-size:12px; margin-bottom:40px;")
        cl.addWidget(sub); cl.addSpacing(32)
        cl.addWidget(section_label("EMAIL ADDRESS")); cl.addSpacing(6)
        self.email_input = QLineEdit(); self.email_input.setPlaceholderText("you@example.com")
        self.email_input.setFixedHeight(42); cl.addWidget(self.email_input); cl.addSpacing(16)
        cl.addWidget(section_label("PASSWORD")); cl.addSpacing(6)
        self.pass_input = QLineEdit(); self.pass_input.setPlaceholderText("••••••••")
        self.pass_input.setEchoMode(QLineEdit.Password); self.pass_input.setFixedHeight(42)
        self.pass_input.returnPressed.connect(self._login)
        cl.addWidget(self.pass_input); cl.addSpacing(28)
        self.login_btn = QPushButton("Sign In"); self.login_btn.setObjectName("primary")
        self.login_btn.setFixedHeight(44); self.login_btn.clicked.connect(self._login)
        cl.addWidget(self.login_btn); cl.addSpacing(16)
        self.err_lbl = QLabel(""); self.err_lbl.setAlignment(Qt.AlignCenter)
        self.err_lbl.setStyleSheet(f"color:{THEME['danger']}; font-size:12px;")
        cl.addWidget(self.err_lbl); cl.addStretch()
        note = QLabel("Contact your administrator to create an account.")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet(f"color:{THEME['subtext']}; font-size:11px;")
        cl.addWidget(note); outer.addWidget(card)

    def _login(self):
        email    = self.email_input.text().strip()
        password = self.pass_input.text()
        if not email or not password:
            self.err_lbl.setText("Please enter email and password."); return
        self.login_btn.setEnabled(False); self.login_btn.setText("Signing in…")
        user = db.login(email, password)
        if user:
            self._app_window = AppWindow(user)
            self._app_window.show(); self.close()
        else:
            self.err_lbl.setText("Invalid email or password.")
            self.login_btn.setEnabled(True); self.login_btn.setText("Sign In")
            self.pass_input.clear()

# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(make_app_icon()))
    win = LoginWindow()
    win.show()
    sys.exit(app.exec_())
