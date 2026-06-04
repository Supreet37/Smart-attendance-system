"""
enroll_voice.py — voice recording + 3-model ensemble feature extraction
FFT (baseline) + Resemblyzer (GE2E) + SpeechBrain (ECAPA-TDNN)
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

# ── Audio ─────────────────────────────────────────────────
try:
    import sounddevice as sd
    from scipy.io import wavfile
    AUDIO_AVAILABLE = True
except Exception:
    sd = None
    wavfile = None
    AUDIO_AVAILABLE = False

# ── Resemblyzer ───────────────────────────────────────────
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pathlib import Path
    _resemblyzer_encoder = VoiceEncoder()
    RESEMBLYZER_OK = True
    print("[ok] Resemblyzer loaded")
except Exception as e:
    _resemblyzer_encoder = None
    RESEMBLYZER_OK = False
    print(f"[warn] Resemblyzer not available: {e}")

# ── SpeechBrain ───────────────────────────────────────────
try:
    import speechbrain.inference as _sb_p
    import importlib, types
    _SBRec = getattr(_sb_p, "SpeakerRecognition", None)
    if _SBRec is None:
        import speechbrain.inference as _sb_inf
        _SBRec = _sb_inf.speaker.SpeakerRecognition
    _speechbrain_model = _SBRec.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    SPEECHBRAIN_OK = True
    print("[ok] SpeechBrain ECAPA loaded")
except Exception as e:
    _speechbrain_model = None
    SPEECHBRAIN_OK = False
    print(f"[warn] SpeechBrain not available: {e}")

BTN_FG = "#FFFFFF"


# ╔══════════════════════════════════════════════════════╗
# ║  Feature Extraction — all 3 models                  ║
# ╚══════════════════════════════════════════════════════╝

def extract_fft_features(audio_int16):
    """Original 26-band FFT feature vector."""
    if audio_int16 is None or len(audio_int16) == 0:
        return np.zeros(26, dtype=np.float32)
    signal = np.asarray(audio_int16, dtype=np.float32).flatten()
    max_val = np.max(np.abs(signal))
    if max_val < 1e-6:
        return np.zeros(26, dtype=np.float32)
    signal /= (max_val + 1e-6)
    frame_len = int(SAMPLE_RATE * 0.02)
    frames = [signal[i:i + frame_len]
              for i in range(0, len(signal) - frame_len, frame_len // 2)
              if len(signal[i:i + frame_len]) == frame_len]
    if not frames:
        return np.zeros(26, dtype=np.float32)
    window = np.hanning(frame_len)
    energies = [np.abs(np.fft.rfft(f * window)) for f in frames]
    avg_spec = np.mean(energies, axis=0)
    n_bins = len(avg_spec)
    edges = np.logspace(np.log10(1), np.log10(max(n_bins, 2)), 27).astype(int)
    edges = np.clip(edges, 0, n_bins - 1)
    bands = []
    for i in range(26):
        s, e = edges[i], edges[i + 1]
        bands.append(float(np.mean(avg_spec[s:e])) if s < e else 0.0)
    bands = np.array(bands, dtype=np.float32)
    norm = np.linalg.norm(bands)
    return (bands / norm).astype(np.float32) if norm > 1e-6 else np.zeros(26, dtype=np.float32)


def extract_resemblyzer_features(wav_path):
    """256-dim GE2E speaker embedding via Resemblyzer."""
    if not RESEMBLYZER_OK or not os.path.isfile(wav_path):
        return None
    try:
        wav = preprocess_wav(Path(wav_path))
        embedding = _resemblyzer_encoder.embed_utterance(wav)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"[warn] Resemblyzer extraction failed: {e}")
        return None


def extract_speechbrain_features(wav_path):
    """192-dim ECAPA-TDNN speaker embedding via SpeechBrain."""
    if not SPEECHBRAIN_OK or not os.path.isfile(wav_path):
        return None
    try:
        import torchaudio
        import torch
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        with torch.no_grad():
            embedding = _speechbrain_model.encode_batch(waveform)
        return embedding.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print(f"[warn] SpeechBrain extraction failed: {e}")
        return None


def average_fft_features(audio1, audio2=None):
    """Average FFT features from 1 or 2 recordings."""
    f1 = extract_fft_features(audio1)
    if audio2 is not None:
        f2 = extract_fft_features(audio2)
        feat = (f1 + f2) / 2.0
        norm = np.linalg.norm(feat)
        return (feat / norm).astype(np.float32) if norm > 1e-6 else None
    return f1 if np.linalg.norm(f1) > 1e-6 else None


def save_all_embeddings(student_dir, audio1, audio2, wav_path1, wav_path2=None):
    """
    Extract and save all 3 model features.
    Returns dict of what was saved successfully.
    """
    saved = {}

    # 1. FFT
    feat_fft = average_fft_features(audio1, audio2)
    if feat_fft is not None:
        np.save(os.path.join(student_dir, "voice_features.npy"), feat_fft)
        saved["fft"] = True
        print("[ok] FFT features saved")

    # 2. Resemblyzer — use first WAV, or average both if available
    if RESEMBLYZER_OK and wav_path1:
        e1 = extract_resemblyzer_features(wav_path1)
        if wav_path2:
            e2 = extract_resemblyzer_features(wav_path2)
            if e1 is not None and e2 is not None:
                avg = (e1 + e2) / 2.0
                avg /= (np.linalg.norm(avg) + 1e-6)
                np.save(os.path.join(student_dir,
                        "voice_embed_resemblyzer.npy"), avg)
                saved["resemblyzer"] = True
                print("[ok] Resemblyzer embedding saved")
        elif e1 is not None:
            np.save(os.path.join(student_dir,
                    "voice_embed_resemblyzer.npy"), e1)
            saved["resemblyzer"] = True

    # 3. SpeechBrain
    if SPEECHBRAIN_OK and wav_path1:
        e1 = extract_speechbrain_features(wav_path1)
        if wav_path2:
            e2 = extract_speechbrain_features(wav_path2)
            if e1 is not None and e2 is not None:
                avg = (e1 + e2) / 2.0
                avg /= (np.linalg.norm(avg) + 1e-6)
                np.save(os.path.join(student_dir,
                        "voice_embed_speechbrain.npy"), avg)
                saved["speechbrain"] = True
                print("[ok] SpeechBrain embedding saved")
        elif e1 is not None:
            np.save(os.path.join(student_dir,
                    "voice_embed_speechbrain.npy"), e1)
            saved["speechbrain"] = True

    return saved


# ╔══════════════════════════════════════════════════════╗
# ║  Recording                                          ║
# ╚══════════════════════════════════════════════════════╝

def record_and_save(student_dir, attempt=1):
    if not AUDIO_AVAILABLE:
        return None, None
    try:
        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        if recording is None or len(recording) == 0:
            return None, None
        if np.max(np.abs(recording.astype(np.float32))) < 300:
            return None, None
        fname = "voice.wav" if attempt == 1 else "voice2.wav"
        filepath = os.path.join(student_dir, fname)
        wavfile.write(filepath, SAMPLE_RATE, recording)
        return recording, filepath
    except Exception as e:
        print(f"[error] Recording failed: {e}")
        return None, None


# ╔══════════════════════════════════════════════════════╗
# ║  VoiceWidget — Tkinter UI                           ║
# ╚══════════════════════════════════════════════════════╝

class VoiceWidget:
    def __init__(self, parent, student_name, student_dir, on_done):
        self.parent       = parent
        self.student_name = student_name
        self.student_dir  = student_dir
        self.on_done      = on_done
        self.audio1       = None
        self.audio2       = None
        self.wav_path1    = None
        self.wav_path2    = None
        self.attempt      = 0
        self._build()

    def _build(self):
        phrase = VOICE_PHRASE.format(name=self.student_name)
        tk.Label(self.parent, text="🎤  Voice Enrollment",
                 font=FONT_HEAD, bg=BG2, fg=ACCENT).pack(pady=(0, 6))
        tk.Label(self.parent,
                 text="Say the following phrase clearly when prompted:",
                 font=FONT_SMALL, bg=BG2, fg=MUTED).pack()

        phrase_box = tk.Frame(self.parent, bg=BG3,
                              highlightbackground=BORDER, highlightthickness=1)
        phrase_box.pack(fill="x", padx=24, pady=10)
        tk.Label(phrase_box, text=f'"{phrase}"',
                 font=FONT_BODY, bg=BG3, fg=TEXT,
                 wraplength=500, justify="center").pack(padx=16, pady=12)

        # Model status indicators
        status_row = tk.Frame(self.parent, bg=BG2)
        status_row.pack(pady=(0, 6))
        for label, ok in [("FFT", True),
                          ("Resemblyzer", RESEMBLYZER_OK),
                          ("SpeechBrain", SPEECHBRAIN_OK)]:
            col  = ACCENT2 if ok else MUTED
            text = f"✓ {label}" if ok else f"○ {label}"
            tk.Label(status_row, text=text, font=FONT_SMALL,
                     bg=BG2, fg=col, padx=8).pack(side="left")

        self.status = tk.Label(self.parent,
                               text="Press RECORD when ready",
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
            self.rec_btn.config(state="disabled",
                                text="No audio", bg=BG3, fg=MUTED)
            self.status.config(
                text="sounddevice/scipy not installed", fg=WARN)

    def _start_record(self):
        self.attempt += 1
        label = ("Recording attempt 1 — speak now…"
                 if self.attempt == 1
                 else "Recording attempt 2 — speak again…")
        self.status.config(text=label, fg=WARN)
        self.rec_btn.config(state="disabled", text="Recording…")
        self._animate(0)
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        audio, path = record_and_save(self.student_dir, attempt=self.attempt)
        self.parent.after(0, lambda: self._on_record_done(audio, path))

    def _on_record_done(self, audio, path):
        if audio is None:
            self.status.config(
                text="Recording failed — try again", fg=WARN)
            self.rec_btn.config(state="normal", text="⏺  RECORD")
            self.attempt -= 1
            return

        if self.attempt == 1:
            self.audio1    = audio
            self.wav_path1 = path
            self.status.config(
                text="✓ First recording saved. Record once more for accuracy.",
                fg=ACCENT2)
            self.rec_btn.config(state="normal", text="⏺  RECORD AGAIN")
        else:
            self.audio2    = audio
            self.wav_path2 = path
            self.status.config(
                text="⏳ Extracting features from all models…", fg=ACCENT)
            self.rec_btn.config(state="disabled",
                                text="Processing…", bg=BG3, fg=MUTED)
            # Run heavy extraction off the main thread
            threading.Thread(target=self._extract_thread, daemon=True).start()

    def _extract_thread(self):
        saved = save_all_embeddings(
            self.student_dir,
            self.audio1, self.audio2,
            self.wav_path1, self.wav_path2
        )
        self.parent.after(0, lambda: self._on_extract_done(saved))

    def _on_extract_done(self, saved):
        models_ready = ", ".join(k.upper() for k in saved)
        self.status.config(
            text=f"✅  Enrolled with: {models_ready}",
            fg=ACCENT2)
        self.skip_btn.config(
            text="FINISH  ✓", bg=ACCENT2,
            font=FONT_BTN,
            command=lambda: self.on_done(saved))

    def _animate(self, step):
        total = int(RECORD_SECONDS / 0.05)
        prog  = min(step / total, 1.0)
        w     = int(420 * prog)
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, 420, 10, fill=BG3, outline="")
        if w > 0:
            self.bar.create_rectangle(0, 0, w, 10, fill=WARN, outline="")
        if prog < 1.0 and self.rec_btn.winfo_exists():
            self.parent.after(50, self._animate, step + 1)