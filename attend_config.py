"""
attend_config.py — shared constants for the attendance system
Professional white + navy blue theme
"""

import os
import platform

# ── Directories ───────────────────────────────────────────
STUDENTS_DIR   = "students"
ATTENDANCE_DIR = "attendance"

# ── Pipeline step IDs ────────────────────────────────────
STEPS = ["face", "hand", "liveness", "voice"]

STEP_LABELS = {
    "face":     "Step 1 — Face Recognition",
    "hand":     "Step 2 — Raise Your Hand",
    "liveness": "Step 3 — Blink & Nod",
    "voice":    "Step 4 — Voice Verification",
}

STEP_INSTRUCTIONS = {
    "face":     "Look straight at the camera",
    "hand":     "Raise your hand above your shoulder  ✋",
    "liveness": "Blink once, then slowly nod your head",
    "voice":    "Say your enrollment phrase when the timer starts",
}

# ── Face recognition ──────────────────────────────────────
FACE_MATCH_FRAMES   = 10
FACE_HIST_BINS      = 32
FACE_MATCH_THRESH   = 0.50       # lowered slightly for real-world tolerance

# ── Hand detection ────────────────────────────────────────
HAND_CONFIRM_FRAMES = 8
HAND_MIN_AREA       = 2800
HAND_UPPER_FRAC     = 0.55

# ── Liveness ─────────────────────────────────────────────
BLINK_EAR_THRESH    = 0.22
BLINK_CONSEC_FRAMES = 1
NOD_SHIFT_PX        = 18
NOD_WINDOW_S        = 3.0

# ── Voice ─────────────────────────────────────────────────
SAMPLE_RATE         = 44100
RECORD_SECONDS      = 3
VOICE_MATCH_THRESH  = 0.40       # lowered for better pass rate
VOICE_MAX_RETRIES   = 3

# ── Voice Ensemble ────────────────────────────────────────
VOICE_ENSEMBLE_WEIGHTS = {
    "fft":         0.20,
    "resemblyzer": 0.50,
    "speechbrain": 0.30,
}

VOICE_FFT_THRESH         = 0.40
VOICE_RESEMBLYZER_THRESH = 0.75
VOICE_SPEECHBRAIN_THRESH = 0.70
VOICE_ENSEMBLE_THRESH    = 0.68   # weighted score must exceed this to pass

# ── GPS ───────────────────────────────────────────────────
GPS_ENABLED         = True       # set False to skip GPS lookup
GPS_TIMEOUT_S       = 5          # seconds to wait for location

# ── Platform ─────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"
IS_MAC     = platform.system() == "Darwin"

# ── Camera ────────────────────────────────────────────────
CAMERA_INDEX        = 0          # change to 1/2 if camera 0 doesn't work

# ── Professional White + Navy Blue Palette ────────────────
BG          = "#F0F4F8"
BG2         = "#FFFFFF"
BG3         = "#E8EEF4"
NAVY        = "#1B2A4A"
ACCENT      = "#1D4ED8"
ACCENT2     = "#16A34A"
WARN        = "#DC2626"
TEXT        = "#1B2A4A"
MUTED       = "#64748B"
BORDER      = "#CBD5E1"
YELLOW      = "#D97706"
STEP_DONE   = "#16A34A"
STEP_TODO   = "#CBD5E1"
STEP_NOW    = "#1D4ED8"
HDR_BG      = "#1B2A4A"
HDR_TEXT    = "#FFFFFF"
BTN_FG      = "#FFFFFF"

# ── Fonts ─────────────────────────────────────────────────
FONT_HEAD   = ("Segoe UI", 16, "bold")
FONT_BODY   = ("Segoe UI", 12)
FONT_SMALL  = ("Segoe UI", 10)
FONT_BTN    = ("Segoe UI", 11, "bold")
FONT_STEP   = ("Segoe UI", 9,  "bold")
FONT_LARGE  = ("Segoe UI", 20, "bold")
FONT_MONO   = ("Segoe UI", 11)
