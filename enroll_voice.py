"""
enroll_voice.py — voice recording + MFCC-band feature extraction
"""

import os
import threading
import numpy as np
import tkinter as tk
from enroll_config import (
    SAMPLE_RATE, RECORD_SECONDS, STUDENTS_DIR, BG2, BG3, ACCENT,
    ACCENT2, WARN, TEXT, MUTED, BORDER, FONT_HEAD, FONT_BODY,
    FONT_SMALL, FONT_BTN, VOICE_PHRASE
)

try:
    import sounddevice as sd
    from scipy.io import wavfile
    AUDIO_AVAILABLE = True
except Exception:
    sd = None
    wavfile = None
    AUDIO_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────
BTN_FG = "#FFFFFF"


def extract_voice_features(audio_int16):
    """Extract 26-band FFT energy features (MFCC-style)."""
    # Safety check for empty/invalid audio
    if audio_int16 is None or len(audio_int16) == 0:
        return np.zeros(26, dtype=np.float32)
    
    signal = np.asarray(audio_int16, dtype=np.float32).flatten()
    max_val = np.max(np.abs(signal))
    if max_val < 1e-6 or not np.isfinite(max_val):
        return np.zeros(26, dtype=np.float32)
    
    signal /= (max_val + 1e-6)

    frame_len = int(SAMPLE_RATE * 0.02)
    frames = [signal[i:i + frame_len]
              for i in range(0, len(signal) - frame_len, frame_len // 2)
              if len(signal[i:i + frame_len]) == frame_len]
    
    # Safety check for no frames
    if not frames:
        return np.zeros(26, dtype=np.float32)

    window = np.hanning(frame_len)
    energies = []
    for f in frames:
        spec = np.abs(np.fft.rfft(f * window))
        energies.append(spec)

    avg_spec = np.nan_to_num(np.mean(energies, axis=0))
    n_bins = len(avg_spec)
    edges = np.logspace(np.log10(1), np.log10(max(n_bins, 2)), 27).astype(int)
    edges = np.clip(edges, 0, n_bins - 1)
    
    bands = []
    for i in range(26):
        start = edges[i]
        end = edges[i + 1]
        if start >= n_bins or start == end:
            bands.append(0.0)
        else:
            band_val = np.mean(avg_spec[start:end])
            bands.append(band_val if not np.isnan(band_val) else 0.0)
    
    bands = np.array(bands, dtype=np.float32)
    norm = np.linalg.norm(bands)
    if norm < 1e-6 or not np.isfinite(norm):
        return np.zeros(26, dtype=np.float32)
    return np.nan_to_num(bands / norm).astype(np.float32)


def record_and_save(student_dir, attempt=1):
    """Record audio and save as WAV file."""
    if not AUDIO_AVAILABLE:
        return None, None
    try:
        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        
        # Check if recording has actual audio (not just silence)
        if recording is None or len(recording) == 0:
            return None, None
        if np.max(np.abs(recording.astype(np.float32))) < 300:
            return None, None
            
        fname = "voice.wav" if attempt == 1 else "voice2.wav"
        filepath = os.path.join(student_dir, fname)
        wavfile.write(filepath, SAMPLE_RATE, recording)
        return recording, filepath
    except Exception as e:
        print(f"Recording error: {e}")
        return None, None


def average_voice_features(audio1, audio2=None):
    """Average features from 1 or 2 recordings and save."""
    f1 = extract_voice_features(audio1)
    if audio2 is not None:
        f2 = extract_voice_features(audio2)
        feat = (f1 + f2) / 2.0
        norm = np.linalg.norm(feat)
        feat = feat / norm if norm > 1e-6 and np.isfinite(norm) else np.zeros(26, dtype=np.float32)
    else:
        feat = f1
    feat = np.nan_to_num(feat).astype(np.float32)
    return feat if np.linalg.norm(feat) > 1e-6 else None


class VoiceWidget:
    """Voice recording widget for enrollment."""

    def __init__(self, parent, student_name, student_dir, on_done):
        self.parent = parent
        self.student_name = student_name
        self.student_dir = student_dir
        self.on_done = on_done
        self.audio1 = None
        self.audio2 = None
        self.attempt = 0
        self._build()

    def _build(self):
        phrase = VOICE_PHRASE.format(name=self.student_name)

        tk.Label(self.parent, text="🎤  Voice Enrollment",
                 font=FONT_HEAD, bg=BG2, fg=ACCENT).pack(pady=(0, 6))

        tk.Label(self.parent, text="Say the following phrase clearly when prompted:",
                 font=FONT_SMALL, bg=BG2, fg=MUTED).pack()

        phrase_box = tk.Frame(self.parent, bg=BG3,
                              highlightbackground=BORDER, highlightthickness=1)
        phrase_box.pack(fill="x", padx=24, pady=10)
        tk.Label(phrase_box, text=f'"{phrase}"',
                 font=FONT_BODY, bg=BG3, fg=TEXT,
                 wraplength=500, justify="center").pack(padx=16, pady=12)

        self.status = tk.Label(self.parent, text="Press RECORD when ready",
                               font=FONT_BODY, bg=BG2, fg=MUTED)
        self.status.pack(pady=(4, 2))

        self.bar = tk.Canvas(self.parent, width=420, height=10,
                             bg=BG3, highlightthickness=0)
        self.bar.pack(pady=(2, 10))

        btn_row = tk.Frame(self.parent, bg=BG2)
        btn_row.pack(pady=4)

        self.rec_btn = tk.Button(
            btn_row, text="⏺  RECORD",
            font=FONT_BTN, bg=ACCENT, fg=BTN_FG,
            relief="flat", padx=20, pady=10, cursor="hand2",
            command=self._start_record)
        self.rec_btn.pack(side="left", padx=8)

        self.skip_btn = tk.Button(
            btn_row, text="Skip voice  →",
            font=FONT_SMALL, bg=BG3, fg=MUTED,
            relief="flat", padx=14, pady=10, cursor="hand2",
            command=lambda: self.on_done(None))
        self.skip_btn.pack(side="left", padx=8)

        if not AUDIO_AVAILABLE:
            self.rec_btn.config(state="disabled", text="No audio", bg=BG3, fg=MUTED)
            self.status.config(text="sounddevice/scipy not installed", fg=WARN)

    def _start_record(self):
        self.attempt += 1
        label = "Recording attempt 1 — speak now…" if self.attempt == 1 \
            else "Recording attempt 2 — speak again…"
        self.status.config(text=label, fg=WARN)
        self.rec_btn.config(state="disabled", text="Recording…")
        self._animate(0)
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        audio, path = record_and_save(self.student_dir, attempt=self.attempt)
        self.parent.after(0, lambda: self._on_record_done(audio))

    def _on_record_done(self, audio):
        if audio is None:
            self.status.config(text="Recording failed — try again", fg=WARN)
            self.rec_btn.config(state="normal", text="⏺  RECORD")
            self.attempt -= 1  # Decrement to allow retry
            return

        if self.attempt == 1:
            self.audio1 = audio
            self.status.config(
                text="✓  First recording saved. Record once more for accuracy.",
                fg=ACCENT2)
            self.rec_btn.config(state="normal", text="⏺  RECORD AGAIN")
        else:
            self.audio2 = audio
            feat = average_voice_features(self.audio1, self.audio2)
            if feat is None:
                self.status.config(text="Voice sample too quiet - try again", fg=WARN)
                self.rec_btn.config(state="normal", text="RECORD AGAIN")
                self.attempt = 1
                return
            save_path = os.path.join(self.student_dir, "voice_features.npy")
            np.save(save_path, feat)
            self.status.config(text="✅  Voice enrolled (2 samples averaged)!", fg=ACCENT2)
            self.rec_btn.config(state="disabled", text="✓  Done", bg=BG3, fg=MUTED)
            self.skip_btn.config(text="FINISH  ✓", bg=ACCENT2, font=FONT_BTN)
            self.skip_btn.config(command=lambda: self.on_done(feat))

    def _animate(self, step):
        total = int(RECORD_SECONDS / 0.05)
        prog = min(step / total, 1.0)
        w = int(420 * prog)
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, 420, 10, fill=BG3, outline="")
        if w > 0:
            self.bar.create_rectangle(0, 0, w, 10, fill=WARN, outline="")
        if prog < 1.0 and self.rec_btn.winfo_exists():
            self.parent.after(50, self._animate, step + 1)
