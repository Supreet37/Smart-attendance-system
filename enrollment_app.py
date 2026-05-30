"""
enrollment_app.py — main enrollment window
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import time
import threading
from datetime import datetime

from enroll_config import (
    STUDENTS_DIR, POSES, BG, BG2, BG3, ACCENT, ACCENT2, WARN,
    TEXT, MUTED, BORDER, HDR_BG, HDR_TEXT, BTN_FG, FONT_HEAD,
    FONT_BODY, FONT_SMALL, FONT_BTN, FONT_STEP, WIN_WIDTH, WIN_HEIGHT,
    STEP_DONE, STEP_TODO, STEP_NOW
)
from enroll_camera import CameraCapture, detect_face
from enroll_voice import VoiceWidget


class EnrollmentApp(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.pack(fill="both", expand=True)
        self.parent = parent

        self.student_name = tk.StringVar()
        self.roll_number = tk.StringVar()
        self.student_dir = ""
        self.captured_photos = []
        self.camera = None

        self._build_ui()
        self._show_confirmation()

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=HDR_BG, pady=0)
        header.pack(fill="x")
        logo = tk.Frame(header, bg=ACCENT, padx=14, pady=10)
        logo.pack(side="left")
        tk.Label(logo, text="◈", font=("Segoe UI", 14, "bold"),
                 bg=ACCENT, fg=HDR_TEXT).pack()
        tk.Label(header, text="ENROLLMENT SYSTEM",
                 font=FONT_HEAD, bg=HDR_BG, fg=HDR_TEXT).pack(side="left", padx=16)

        # Close button
        close_btn = tk.Button(header, text="✕", font=("Segoe UI", 12),
                              bg=HDR_BG, fg=HDR_TEXT, relief="flat",
                              padx=12, pady=8, cursor="hand2",
                              command=self._on_close)
        close_btn.pack(side="right")
        close_btn.bind("<Enter>", lambda e: close_btn.config(bg=WARN))
        close_btn.bind("<Leave>", lambda e: close_btn.config(bg=HDR_BG))

        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x")
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(fill="both", expand=True, padx=20, pady=20)

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _show_confirmation(self):
        self._clear_content()
        # Form card
        card = tk.Frame(self.content, bg=BG2, padx=40, pady=36,
                        highlightbackground=BORDER, highlightthickness=1)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(card, text="NEW STUDENT ENROLLMENT",
                 font=("Segoe UI", 14, "bold"), bg=BG2, fg=ACCENT).pack(pady=(0, 4))
        tk.Label(card, text="Fill in the details below to begin",
                 font=FONT_SMALL, bg=BG2, fg=MUTED).pack(pady=(0, 28))

        # Name field
        self._field(card, "Full Name", self.student_name, "e.g. Ravi Kumar")
        tk.Frame(card, bg=BG, height=14).pack()
        self._field(card, "Roll Number", self.roll_number, "e.g. CSE2024001")
        tk.Frame(card, bg=BG, height=30).pack()

        btn = tk.Button(card, text="START ENROLLMENT  ▶",
                        font=FONT_BTN, bg=ACCENT, fg=BTN_FG,
                        relief="flat", padx=28, pady=12, cursor="hand2",
                        command=self._start_enrollment)
        btn.pack()
        btn.bind("<Enter>", lambda e: btn.config(bg="#3b82f6"))
        btn.bind("<Leave>", lambda e: btn.config(bg=ACCENT))

    def _field(self, parent, label, var, placeholder):
        tk.Label(parent, text=label.upper(), font=FONT_SMALL,
                 bg=BG2, fg=MUTED).pack(anchor="w")
        entry = tk.Entry(parent, textvariable=var, font=FONT_BODY,
                         bg=BG3, fg=TEXT, insertbackground=TEXT,
                         relief="flat", width=32,
                         highlightbackground=BORDER, highlightthickness=1)
        entry.pack(fill="x", ipady=8, pady=(4, 0))
        return entry

    def _start_enrollment(self):
        name = self.student_name.get().strip()
        roll = self.roll_number.get().strip()
        if not name or not roll:
            messagebox.showwarning("Missing Info", "Please fill in both fields.")
            return

        folder = os.path.join(STUDENTS_DIR, f"{roll}_{name.replace(' ', '_')}")
        if os.path.exists(folder):
            if not messagebox.askyesno("Already Enrolled",
                                       f"{name} ({roll}) already exists.\nOverwrite?"):
                return
            import shutil
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        self.student_dir = folder
        self.captured_photos = []
        self._show_camera()

    def _show_camera(self):
        self._clear_content()
        self._build_camera_ui()

        self.camera = CameraCapture(
            on_captured=self._on_photo_captured,
            on_frame=self._update_camera_frame,
            on_error=lambda e: messagebox.showerror("Camera Error", e)
        )
        self.camera.start()

    def _build_camera_ui(self):
        left = tk.Frame(self.content, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(self.content, bg=BG, width=240)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        cam_border = tk.Frame(left, bg=BG3, padx=2, pady=2,
                              highlightbackground=BORDER, highlightthickness=1)
        cam_border.pack()
        self.cam_label = tk.Label(cam_border, bg="#000000")
        self.cam_label.pack()

        self.inst_label = tk.Label(left, text="", font=FONT_STEP,
                                   bg=BG, fg=ACCENT, wraplength=480)
        self.inst_label.pack(pady=(10, 4))
        self.sub_label = tk.Label(left, text="", font=FONT_SMALL, bg=BG, fg=MUTED)
        self.sub_label.pack()

        # Manual capture button
        self.manual_btn = tk.Button(left, text="📸  CAPTURE MANUALLY",
                                    font=FONT_BTN, bg=BG3, fg=TEXT,
                                    relief="flat", padx=20, pady=8, cursor="hand2",
                                    command=self._manual_capture)
        self.manual_btn.pack(pady=8)

        # Progress steps
        tk.Label(right, text="CAPTURE PROGRESS",
                 font=FONT_STEP, bg=BG, fg=MUTED).pack(anchor="w", pady=(0, 10))
        self.step_frames = []
        for i, (pid, pdesc, _) in enumerate(POSES):
            sf = tk.Frame(right, bg=BG2, padx=12, pady=8,
                          highlightbackground=STEP_TODO, highlightthickness=1)
            sf.pack(fill="x", pady=3)
            dot = tk.Label(sf, text="○", font=("Courier New", 14), bg=BG2, fg=STEP_TODO)
            dot.pack(side="left")
            info_f = tk.Frame(sf, bg=BG2)
            info_f.pack(side="left", padx=8)
            num_lbl = tk.Label(info_f, text=f"Pose {i+1}", font=FONT_STEP, bg=BG2, fg=MUTED)
            num_lbl.pack(anchor="w")
            desc_lbl = tk.Label(info_f, text=pdesc, font=FONT_SMALL, bg=BG2, fg=MUTED)
            desc_lbl.pack(anchor="w")
            self.step_frames.append({"frame": sf, "dot": dot, "num": num_lbl, "desc": desc_lbl})

    def _update_step_ui(self):
        if not self.camera:
            return
        for i, sf in enumerate(self.step_frames):
            if i < self.camera.pose_index:
                sf["frame"].config(highlightbackground=STEP_DONE)
                sf["dot"].config(text="✓", fg=STEP_DONE)
            elif i == self.camera.pose_index:
                sf["frame"].config(highlightbackground=STEP_NOW)
                sf["dot"].config(text="►", fg=STEP_NOW)
            else:
                sf["frame"].config(highlightbackground=STEP_TODO)
                sf["dot"].config(text="○", fg=STEP_TODO)

    def _update_camera_frame(self, display, face, gate_ok, hint):
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self.cam_label.config(image=photo)
        self.cam_label.image = photo

        if self.camera and self.camera.pose_index < len(POSES):
            _, pdesc, _ = POSES[self.camera.pose_index]
            self.inst_label.config(text=f"Pose {self.camera.pose_index + 1}: {pdesc}")
            self.sub_label.config(text=hint, fg=ACCENT2 if gate_ok else WARN)

    def _on_photo_captured(self, frame, pose_id):
        filepath = os.path.join(self.student_dir, f"{pose_id}.jpg")
        cv2.imwrite(filepath, frame)
        self.captured_photos.append(filepath)
        self._update_step_ui()

        if self.camera and self.camera.pose_index >= len(POSES):
            self.camera.stop()
            self._show_voice()

    def _manual_capture(self):
        if self.camera:
            self.camera.manual_capture()

    def _show_voice(self):
        self._clear_content()
        voice_frame = tk.Frame(self.content, bg=BG)
        voice_frame.place(relx=0.5, rely=0.5, anchor="center")

        def on_voice_done(feat):
            self._finish_enrollment()

        self.voice_widget = VoiceWidget(
            voice_frame, self.student_name.get(),
            self.student_dir, on_voice_done
        )

    def _finish_enrollment(self):
        meta = {
            "name": self.student_name.get().strip(),
            "roll_number": self.roll_number.get().strip(),
            "enrolled_at": datetime.now().isoformat(),
            "photos": [os.path.basename(p) for p in self.captured_photos],
        }
        with open(os.path.join(self.student_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._show_done(meta)

    def _show_done(self, meta):
        self._clear_content()
        card = tk.Frame(self.content, bg=BG2, padx=48, pady=40,
                        highlightbackground=ACCENT2, highlightthickness=2)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(card, text="✅  ENROLLMENT COMPLETE",
                 font=("Segoe UI", 16, "bold"), bg=BG2, fg=ACCENT2).pack(pady=(0, 8))

        info_box = tk.Frame(card, bg=BG3, padx=20, pady=14,
                            highlightbackground=BORDER, highlightthickness=1)
        info_box.pack(fill="x", pady=(8, 20))

        rows = [("Name", meta["name"]), ("Roll No", meta["roll_number"]),
                ("Photos", f"{len(meta['photos'])} captured"), ("Saved to", self.student_dir)]
        for label, val in rows:
            row = tk.Frame(info_box, bg=BG3)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{label:<10}", font=FONT_SMALL, bg=BG3, fg=MUTED).pack(side="left")
            tk.Label(row, text=val, font=FONT_BODY, bg=BG3, fg=TEXT).pack(side="left", padx=10)

        btn = tk.Button(card, text="ENROLL ANOTHER  ➕",
                        font=FONT_BTN, bg=ACCENT, fg=BTN_FG,
                        relief="flat", padx=22, pady=10, cursor="hand2",
                        command=self._reset)
        btn.pack()

    def _reset(self):
        self.student_name.set("")
        self.roll_number.set("")
        self.captured_photos = []
        self._show_confirmation()

    def _on_close(self):
        if self.camera:
            self.camera.stop()
        self.parent.destroy()

if __name__ == "__main__":
    from enroll_config import WIN_WIDTH, WIN_HEIGHT
    root = tk.Tk()
    root.title("Smart Attendance — Enrollment")
    root.resizable(False, False)
    root.configure(bg="#F0F4F8")
    sx = root.winfo_screenwidth()
    sy = root.winfo_screenheight()
    root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{(sx-WIN_WIDTH)//2}+{(sy-WIN_HEIGHT)//2}")
    EnrollmentApp(root)
    root.mainloop()