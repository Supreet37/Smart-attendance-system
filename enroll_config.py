"""
enroll_config.py — shared constants for enrollment
"""

import os

# ── Directories ──────────────────────────────────────────
STUDENTS_DIR = "students"

# ── Voice ────────────────────────────────────────────────
VOICE_PHRASE   = "My name is {name} and I am present"
SAMPLE_RATE    = 44100
RECORD_SECONDS = 3

# ── Face capture ─────────────────────────────────────────
FACE_HOLD_TIME  = 1.0
POSE_SHIFT_PX   = 22
SMILE_RATIO_MIN = 0.18

# ── 9 poses ──────────────────────────────────────────────
POSES = [
    ("front_1",     "Look straight at the camera",        "centre"),
    ("front_2",     "Stay straight — second shot",        "centre"),
    ("left",        "Turn your head slightly LEFT",       "left"),
    ("right",       "Turn your head slightly RIGHT",      "right"),
    ("up",          "Tilt your head slightly UP",         "up"),
    ("down",        "Tilt your head slightly DOWN",       "down"),
    ("smile",       "Give a natural SMILE  ☺",            "smile"),
    ("left_extra",  "Turn a little more to the LEFT",     "left"),
    ("right_extra", "Turn a little more to the RIGHT",    "right"),
]

UPLOAD_LABELS = [
    ("front_1",     "Front — neutral"),
    ("front_2",     "Front — second"),
    ("left",        "Left turn"),
    ("right",       "Right turn"),
    ("up",          "Head up"),
    ("down",        "Head down"),
    ("smile",       "Smiling"),
    ("left_extra",  "Far left"),
    ("right_extra", "Far right"),
]

# ── Colours (same palette as attend_config) ───────────────
BG         = "#F0F4F8"
BG2        = "#FFFFFF"
BG3        = "#E8EEF4"
NAVY       = "#1B2A4A"
ACCENT     = "#1D4ED8"
ACCENT2    = "#16A34A"
WARN       = "#DC2626"
TEXT       = "#1B2A4A"
MUTED      = "#64748B"
BORDER     = "#CBD5E1"
STEP_DONE  = "#16A34A"
STEP_TODO  = "#CBD5E1"
STEP_NOW   = "#1D4ED8"
HDR_BG     = "#1B2A4A"
HDR_TEXT   = "#FFFFFF"
BTN_FG     = "#FFFFFF"

# ── Fonts ─────────────────────────────────────────────────
FONT_HEAD  = ("Segoe UI", 16, "bold")
FONT_BODY  = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 10)
FONT_BTN   = ("Segoe UI", 11, "bold")
FONT_STEP  = ("Segoe UI",  9, "bold")

# ── Window ────────────────────────────────────────────────
WIN_WIDTH  = 920
WIN_HEIGHT = 680
