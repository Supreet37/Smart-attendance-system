"""
attend_pipeline.py — sequential 4-step verification state machine
"""

import cv2
import time
import threading
from attend_config import (
    FACE_MATCH_FRAMES, HAND_CONFIRM_FRAMES,
    RECORD_SECONDS, VOICE_MAX_RETRIES
)
from attend_detectors import (
    detect_face, match_face, detect_raised_hand, draw_hand_overlay,
    detect_liveness, record_voice, match_voice, log_attendance, AUDIO_OK
)

# Step ID constants
S_FACE       = "face"
S_HAND_PROMPT = "hand_prompt"
S_HAND       = "hand"
S_LIVENESS   = "liveness"
S_VOICE      = "voice"
S_DONE       = "done"


class VerificationPipeline:
    def __init__(self, students, on_complete, on_update):
        self.students    = students
        self.on_complete = on_complete
        self.on_update   = on_update
        self._cycle_id   = 0
        self.reset()

    def reset(self):
        self.step             = S_FACE
        self.matched_student  = None
        self.done             = False
        self._face_buf        = []
        self._prompt_start    = None
        self._hand_buf        = 0
        self._liveness_start  = None
        self._blink_ok        = False
        self._nodded          = False
        self._liveness_state  = None
        self._voice_retries   = 0
        self._voice_busy      = False
        self._voice_skipped   = False
        self._cycle_id       += 1

    # ── Main per-frame call ───────────────────────────────────────────────
    def process_frame(self, frame_bgr):
        if self.done:
            return self._make_result(frame_bgr, None, None, None, False)

        face       = detect_face(frame_bgr)
        overlay    = None
        hand_kpts  = None
        gate_ok    = False

        # Step 1: FACE RECOGNITION
        if self.step == S_FACE:
            overlay = "Look straight at the camera"
            if face is not None:
                x, y, w, h = face
                student, sim = match_face(frame_bgr[y:y+h, x:x+w], self.students)
                if student:
                    self._face_buf.append(student)
                else:
                    self._face_buf = []
                if len(self._face_buf) >= FACE_MATCH_FRAMES:
                    self.matched_student = self._face_buf[-1]
                    self.step = S_HAND_PROMPT
                    self._prompt_start = time.time()
                    self.on_update(S_HAND_PROMPT,
                                   f"✓ Face matched: {self.matched_student['name']}", "green")
            else:
                self._face_buf = []

        # Step 1b: brief HAND PROMPT banner
        elif self.step == S_HAND_PROMPT:
            overlay = "✋  RAISE YOUR HAND  ✋"
            gate_ok = True
            if time.time() - (self._prompt_start or time.time()) >= 1.5:
                self.step = S_HAND
                self.on_update(S_HAND, "Raise your hand above your shoulder…", "accent")

        # Step 2: HAND GESTURE
        elif self.step == S_HAND:
            is_raised, lms, bbox, wrist_y, shoulder = detect_raised_hand(frame_bgr, face)
            if is_raised:
                self._hand_buf = min(self._hand_buf + 1, HAND_CONFIRM_FRAMES + 5)
                gate_ok  = True
                overlay  = "✓ Hand detected — keep it raised!"
            else:
                self._hand_buf = max(0, self._hand_buf - 2)
                overlay  = "✋  RAISE YOUR HAND ABOVE YOUR SHOULDER  ✋"

            hand_kpts = {
                "bbox": bbox, "shoulder": shoulder, "is_raised": is_raised
            } if bbox is not None else None

            if self._hand_buf >= HAND_CONFIRM_FRAMES:
                self.step = S_LIVENESS
                self._liveness_start = time.time()
                self._liveness_state = None
                self.on_update(S_LIVENESS,
                               "✓ Hand confirmed — Blink once, then nod your head", "green")

        # Step 3: LIVENESS — blink + nod
        elif self.step == S_LIVENESS:
            if face is not None:
                blink, nod, ear, nod_mag, self._liveness_state = detect_liveness(
                    frame_bgr, face, self._liveness_state)
                if blink and not self._blink_ok:
                    self._blink_ok = True
                    self.on_update(S_LIVENESS, "✓ Blink detected — now nod your head", "green")
                if nod and not self._nodded:
                    self._nodded = True
                    self.on_update(S_LIVENESS, "✓ Nod detected!", "green")

                if not self._blink_ok:
                    overlay = "👁  BLINK YOUR EYES  👁"
                elif not self._nodded:
                    overlay = "🙂  NOW NOD YOUR HEAD  🙂"
                else:
                    overlay = "✓ Liveness confirmed!"

            if self._blink_ok and self._nodded:
                self.step = S_VOICE
                self.on_update(S_VOICE, "✓ Liveness confirmed — preparing voice check…", "green")
                threading.Timer(0.8, self._start_voice).start()

        # Step 4: VOICE
        elif self.step == S_VOICE:
            overlay = "🔴  Recording…" if self._voice_busy else "🎤  Speak your phrase now"

        return self._make_result(frame_bgr, face, overlay, hand_kpts, gate_ok)

    # ── Internal helpers ──────────────────────────────────────────────────
    def _make_result(self, frame, face, overlay_text, hand_kpts, gate_ok):
        return type("R", (), {
            "frame": frame, "step": self.step, "face": face,
            "overlay_text": overlay_text, "hand_kpts": hand_kpts,
            "gate_ok": gate_ok, "hand_fingers": 0,
        })()

    def _start_voice(self):
        if self.step != S_VOICE or self._voice_busy or self.done:
            return
        if not AUDIO_OK:
            self._finish(voice_skipped=True)
            return
        self._voice_busy = True
        my_cycle = self._cycle_id
        threading.Thread(target=self._voice_thread, args=(my_cycle,), daemon=True).start()

    def _voice_thread(self, my_cycle):
        audio = record_voice(RECORD_SECONDS)
        if self._cycle_id != my_cycle:
            self._voice_busy = False
            return
        matched, sim = match_voice(audio, self.matched_student)
        self._voice_busy = False
        if self._cycle_id != my_cycle:
            return
        if matched:
            self.on_update(S_VOICE, f"✓ Voice verified (score {sim:.2f})", "green")
            time.sleep(0.4)
            if self._cycle_id == my_cycle:
                self._finish(voice_skipped=False)
        else:
            self._voice_retries += 1
            if self._voice_retries < VOICE_MAX_RETRIES:
                self.on_update(S_VOICE,
                    f"Voice didn't match ({sim:.2f}) — retry {self._voice_retries}/{VOICE_MAX_RETRIES}",
                    "warn")
                time.sleep(1.2)
                if self._cycle_id == my_cycle:
                    self._start_voice()
            else:
                self.on_update(S_VOICE,
                    "Voice check failed — attendance marked (flagged)", "warn")
                time.sleep(0.6)
                if self._cycle_id == my_cycle:
                    self._finish(voice_skipped=True)

    def _finish(self, voice_skipped=False):
        if self.done:
            return
        self.done           = True
        self._voice_skipped = voice_skipped
        self.step           = S_DONE
        self.on_complete(self.matched_student, None, voice_skipped)
