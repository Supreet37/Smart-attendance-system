"""
Smart Attendance System — Main Entry Point
Run:  python attendance_app.py
"""

import os
import csv
import tkinter as tk
from tkinter import messagebox
import time
import threading
import cv2
from datetime import datetime

from attend_config import STUDENTS_DIR, ATTENDANCE_DIR, CAMERA_INDEX
from attend_detectors import load_students, log_attendance, open_camera
from attend_pipeline import VerificationPipeline, S_FACE, S_DONE
from attend_ui import AttendanceUI


def _already_marked_today(roll):
    today    = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    if not os.path.isfile(csv_path):
        return False
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == roll:
                return True
    return False


class AttendanceApp:
    def __init__(self):
        os.makedirs(STUDENTS_DIR, exist_ok=True)
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)

        self.students        = load_students()
        self.cap             = None
        self.camera_running  = False
        self.current_frame   = None
        self.pipeline        = None
        self.ui              = None
        self._marked_today   = set()
        self._session_active = False

        self._start_ui()
        self._start_camera()
        self._start_pipeline()
        self._reload_todays_attendance()

    # ── UI ────────────────────────────────────────────────────────────────
    def _start_ui(self):
        self.ui = AttendanceUI(
            self.students,
            self._reset_pipeline,
            on_session_start=self._on_session_start,
            on_session_stop=self._on_session_stop,
            on_enrollment_closed=self._on_enrollment_closed,
        )
        self.ui.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Camera ────────────────────────────────────────────────────────────
    def _start_camera(self):
        self.cap = open_camera(CAMERA_INDEX)
        if self.cap is None:
            messagebox.showerror(
                "Camera Error",
                "Could not open any camera.\n\n"
                "• Make sure your camera is connected and not used by another app.\n"
                "• Try changing CAMERA_INDEX in attend_config.py (0, 1, 2…).\n"
                "• On Linux: check if you are in the 'video' group."
            )
            self.ui.after(100, self._on_close)
            return
        self.camera_running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            frame = cv2.flip(frame, 1)          # mirror so it feels natural
            self.current_frame = frame.copy()
            self.ui.after(0, self._process_frame, frame)
            time.sleep(0.033)                   # ~30 fps cap
        if self.cap:
            self.cap.release()

    def _process_frame(self, frame):
        if self.pipeline is None:
            return
        if not self._session_active:
            self.ui.update_frame_idle(frame)
            return
        result = self.pipeline.process_frame(frame)
        self.ui.update_frame(frame, result)
        self.ui.update_step(self.pipeline.step, "", "muted")

    # ── Pipeline ─────────────────────────────────────────────────────────
    def _start_pipeline(self):
        self.pipeline = VerificationPipeline(
            self.students,
            on_complete=self._on_verification_complete,
            on_update=self._on_pipeline_update,
            on_face_matched=self._on_face_matched,
        )

    def _reset_pipeline(self):
        if self.pipeline:
            self.pipeline.reset()
        self.ui.refresh_steps(S_FACE)

    def _on_pipeline_update(self, step, msg, tone):
        self.ui.update_step(step, msg, tone)

    # ── Session ───────────────────────────────────────────────────────────

    def _on_enrollment_closed(self):
        self.students = load_students()
        if self.pipeline:
            self.pipeline.students = self.students
        self.ui.students = self.students
        self.ui._update_enrolled_badge()
        self.ui.status_lbl.config(
            text=f"Reloaded — {len(self.students)} student(s) enrolled", fg="#1D4ED8")

    def _on_session_start(self):
        self._session_active = True
        self._reset_pipeline()
        self.ui.set_session_ui(active=True)
        self.ui.status_lbl.config(
            text="Session started — stand in front of the camera", fg="#16A34A")

    def _on_session_stop(self):
        self._session_active = False
        if self.pipeline:
            self.pipeline.reset()
        self.ui.set_session_ui(active=False)
        self.ui.status_lbl.config(
            text="Session stopped — press Start Attendance to begin", fg="#64748B")

    # ── Verification complete ─────────────────────────────────────────────
    def _on_verification_complete(self, student, frame, voice_skipped):
        if not student:
            return
        roll = student["roll"]

        # Duplicate guard
        if roll in self._marked_today or _already_marked_today(roll):
            self._marked_today.add(roll)
            self.ui.show_already_marked(student)
            self.ui.after(3000, self._reset_pipeline)
            return

        now      = datetime.now()
        snap_dir = os.path.join(ATTENDANCE_DIR, now.strftime("%Y-%m-%d"))
        os.makedirs(snap_dir, exist_ok=True)
        photo_path = os.path.join(snap_dir,
                                   f"{roll}_{now.strftime('%H%M%S')}.jpg")
        if self.current_frame is not None:
            cv2.imwrite(photo_path, self.current_frame)

        log_attendance(student, photo_path, voice_skipped)
        self._marked_today.add(roll)

        self.ui.show_marked(student, photo_path, now, voice_skipped)
        self.ui.add_attendee(student, self.current_frame)

        # Auto-reset for next student
        self.ui.after(4000, self._reset_pipeline)


    def _reload_todays_attendance(self):
        today = datetime.now().strftime("%Y-%m-%d")
        csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
        if not os.path.isfile(csv_path):
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] != "Roll":   # skip header
                    roll = row[0]
                    self._marked_today.add(roll)
                    # find the matching student and add to UI
                    for s in self.students:
                        if s["roll"] == roll:
                            self.ui.add_attendee(s, None)
                            break

    def _on_face_matched(self, student):
        roll = student["roll"]
        if roll in self._marked_today or _already_marked_today(roll):
            self._marked_today.add(roll)
            self.ui.show_already_marked(student)
            self.ui.after(3000, self._reset_pipeline)
            return True   # True = block the pipeline from advancing
        return False      # False = proceed normally

    # ── Close ─────────────────────────────────────────────────────────────
    def _on_close(self):
        self.camera_running = False
        time.sleep(0.15)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.ui.destroy()

    def run(self):
        self.ui.mainloop()


if __name__ == "__main__":
    app = AttendanceApp()
    app.run()
