"""
database.py  —  BFRB App Database Module
=========================================
SQLite database with SQLAlchemy-style raw sqlite3 interface.
Tables: Account, Session, DetectionEvent, AccountPreferences

Default admin: admin@bfrb.com / Admin1234!
"""

import sqlite3
import bcrypt
import os
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "bfrb.db"


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, stored: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), stored.encode())
    except Exception:
        return False


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all tables and seed default admin if not exists."""
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS Account (
            user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            last_login    TEXT,
            is_active     INTEGER NOT NULL DEFAULT 1,
            account_type  TEXT    NOT NULL DEFAULT 'user'
        );

        CREATE TABLE IF NOT EXISTS Session (
            session_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL REFERENCES Account(user_id),
            start_time       TEXT    NOT NULL,
            end_time         TEXT,
            duration_seconds INTEGER,
            event_count      INTEGER NOT NULL DEFAULT 0,
            video_source     TEXT    DEFAULT 'webcam'
        );

        CREATE TABLE IF NOT EXISTS DetectionEvent (
            event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       INTEGER NOT NULL REFERENCES Session(session_id),
            timestamp        TEXT    NOT NULL,
            behavior_type    TEXT    NOT NULL,
            confidence_score REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS AccountPreferences (
            preference_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL UNIQUE REFERENCES Account(user_id),
            sensitivity    REAL    NOT NULL DEFAULT 0.6,
            alert_cooldown INTEGER NOT NULL DEFAULT 10,
            session_duration INTEGER NOT NULL DEFAULT 60,
            skeleton_overlay INTEGER NOT NULL DEFAULT 1
        );
    """)

    # Seed default admin
    existing = c.execute(
        "SELECT user_id FROM Account WHERE email = ?", ("admin@bfrb.com",)
    ).fetchone()

    if not existing:
        now = datetime.now().isoformat()
        c.execute(
            """INSERT INTO Account (username, email, password_hash, created_at, account_type)
               VALUES (?, ?, ?, ?, ?)""",
            ("admin", "admin@bfrb.com", _hash_password("Admin1234!"), now, "admin"),
        )
        admin_id = c.lastrowid
        c.execute(
            """INSERT INTO AccountPreferences (user_id) VALUES (?)""",
            (admin_id,),
        )

    conn.commit()
    conn.close()


# =============================================================================
# AUTH
# =============================================================================

def login(email: str, password: str):
    """Returns Account row dict or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM Account WHERE email = ? AND is_active = 1", (email,)
    ).fetchone()
    if row and _verify_password(password, row["password_hash"]):
        conn.execute(
            "UPDATE Account SET last_login = ? WHERE user_id = ?",
            (datetime.now().isoformat(), row["user_id"]),
        )
        conn.commit()
        result = dict(row)
        conn.close()
        return result
    conn.close()
    return None


# =============================================================================
# USER MANAGEMENT
# =============================================================================

def get_all_users():
    conn = get_connection()
    rows = conn.execute(
        "SELECT user_id, username, email, created_at, last_login, is_active, account_type "
        "FROM Account ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_non_admin_users():
    conn = get_connection()
    rows = conn.execute(
        "SELECT user_id, username, email, is_active FROM Account "
        "WHERE account_type = 'user' AND is_active = 1 ORDER BY username"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_user(username: str, email: str, password: str, account_type: str = "user"):
    """Returns (True, user_id) or (False, error_message)."""
    conn = get_connection()
    try:
        now = datetime.now().isoformat()
        c = conn.cursor()
        c.execute(
            """INSERT INTO Account (username, email, password_hash, created_at, account_type)
               VALUES (?, ?, ?, ?, ?)""",
            (username, email, _hash_password(password), now, account_type),
        )
        uid = c.lastrowid
        c.execute("INSERT INTO AccountPreferences (user_id) VALUES (?)", (uid,))
        conn.commit()
        conn.close()
        return True, uid
    except sqlite3.IntegrityError as e:
        conn.close()
        if "username" in str(e):
            return False, "Username already exists."
        if "email" in str(e):
            return False, "Email already exists."
        return False, str(e)


def delete_user(user_id: int):
    conn = get_connection()
    conn.execute("UPDATE Account SET is_active = 0 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# =============================================================================
# SESSIONS
# =============================================================================

def start_session(user_id: int, video_source: str = "webcam") -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO Session (user_id, start_time, video_source) VALUES (?, ?, ?)",
        (user_id, datetime.now().isoformat(), video_source),
    )
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid


def end_session(session_id: int, event_count: int):
    conn = get_connection()
    now = datetime.now().isoformat()
    row = conn.execute(
        "SELECT start_time FROM Session WHERE session_id = ?", (session_id,)
    ).fetchone()
    duration = 0
    if row:
        try:
            start = datetime.fromisoformat(row["start_time"])
            duration = int((datetime.now() - start).total_seconds())
        except Exception:
            pass
    conn.execute(
        """UPDATE Session SET end_time = ?, duration_seconds = ?, event_count = ?
           WHERE session_id = ?""",
        (now, duration, event_count, session_id),
    )
    conn.commit()
    conn.close()


def get_sessions_for_user(user_id: int):
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM Session WHERE user_id = ?
           ORDER BY start_time DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_sessions():
    conn = get_connection()
    rows = conn.execute(
        """SELECT s.*, a.username FROM Session s
           JOIN Account a ON s.user_id = a.user_id
           ORDER BY s.start_time DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
# DETECTION EVENTS
# =============================================================================

def log_detection(session_id: int, behavior_type: str, confidence_score: float):
    conn = get_connection()
    conn.execute(
        """INSERT INTO DetectionEvent (session_id, timestamp, behavior_type, confidence_score)
           VALUES (?, ?, ?, ?)""",
        (session_id, datetime.now().isoformat(), behavior_type, confidence_score),
    )
    conn.commit()
    conn.close()


def get_events_for_session(session_id: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM DetectionEvent WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_events_for_user(user_id: int):
    conn = get_connection()
    rows = conn.execute(
        """SELECT d.* FROM DetectionEvent d
           JOIN Session s ON d.session_id = s.session_id
           WHERE s.user_id = ?
           ORDER BY d.timestamp DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
# PREFERENCES
# =============================================================================

def get_preferences(user_id: int) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM AccountPreferences WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "sensitivity": 0.6,
        "alert_cooldown": 10,
        "session_duration": 60,
        "skeleton_overlay": 1,
    }


def save_preferences(user_id: int, sensitivity: float, alert_cooldown: int,
                     session_duration: int, skeleton_overlay: int):
    conn = get_connection()
    conn.execute(
        """UPDATE AccountPreferences
           SET sensitivity = ?, alert_cooldown = ?, session_duration = ?, skeleton_overlay = ?
           WHERE user_id = ?""",
        (sensitivity, alert_cooldown, session_duration, skeleton_overlay, user_id),
    )
    conn.commit()
    conn.close()


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def export_sessions_csv(user_id=None) -> str:
    """Returns CSV string. If user_id is None, exports all users (admin)."""
    import csv, io
    sessions = get_sessions_for_user(user_id) if user_id else get_all_sessions()
    out = io.StringIO()
    if not sessions:
        return "No sessions found."
    writer = csv.DictWriter(out, fieldnames=sessions[0].keys())
    writer.writeheader()
    writer.writerows(sessions)
    return out.getvalue()


def export_events_csv(user_id=None) -> str:
    import csv, io
    events = get_events_for_user(user_id) if user_id else _get_all_events()
    out = io.StringIO()
    if not events:
        return "No events found."
    writer = csv.DictWriter(out, fieldnames=events[0].keys())
    writer.writeheader()
    writer.writerows(events)
    return out.getvalue()


def _get_all_events():
    conn = get_connection()
    rows = conn.execute(
        """SELECT d.*, a.username FROM DetectionEvent d
           JOIN Session s ON d.session_id = s.session_id
           JOIN Account a ON s.user_id = a.user_id
           ORDER BY d.timestamp DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Init on import
init_db()
