"""
attend_ui.py — professional attendance UI
Cross-platform: works on Windows, Linux, macOS.
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os
from datetime import datetime

from attend_config import (
    BG, BG2, BG3, NAVY, ACCENT, ACCENT2, WARN, TEXT, MUTED, BORDER,
    STEP_DONE, STEP_TODO, STEP_NOW, HDR_BG, HDR_TEXT, BTN_FG,
    FONT_HEAD, FONT_BODY, FONT_SMALL, FONT_BTN, FONT_STEP,
    ATTENDANCE_DIR, IS_WINDOWS,
)
from attend_pipeline import S_FACE, S_HAND, S_LIVENESS, S_VOICE, S_DONE
from attend_detectors import draw_hand_overlay, get_location

STEP_DEFS = [
    (S_FACE,     "Step 1", "Face\nRecognition"),
    (S_HAND,     "Step 2", "Raise\nHand"),
    (S_LIVENESS, "Step 3", "Blink\n& Nod"),
    (S_VOICE,    "Step 4", "Voice\nVerify"),
]
STEP_KEYS = [s[0] for s in STEP_DEFS]

CAM_W, CAM_H = 640, 440          # display resolution


class AttendanceUI(tk.Tk):
    def __init__(self, students, on_pipeline_reset,
                 on_session_start=None, on_session_stop=None,on_enrollment_closed=None):
        super().__init__()
        self.students              = students
        self.on_pipeline_reset     = on_pipeline_reset
        self._on_session_start_cb  = on_session_start
        self._on_session_stop_cb   = on_session_stop
        self.attendees_today       = []
        self._cur_step             = S_FACE
        self._session_active       = False
        self._on_enrollment_closed_cb = on_enrollment_closed

        self.title("Smart Attendance System")
        self.configure(bg=BG)

        # ── Maximise window (cross-platform) ─────────────────────────────
        try:
            if IS_WINDOWS:
                self.state("zoomed")
            else:
                self.attributes("-zoomed", True)   # Linux (Tk 8.6+)
        except Exception:
            self.geometry("1280x800")               # safe fallback

        self._build()
        self._tick_clock()

    # ── Build layout ─────────────────────────────────────────────────────
    def _build(self):
        self._build_titlebar()
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)
        self._build_left(main)
        self._build_centre(main)
        self._build_right(main)

    def _build_titlebar(self):
        tb = tk.Frame(self, bg=HDR_BG)
        tb.pack(fill="x")

        logo = tk.Frame(tb, bg=ACCENT, padx=16, pady=10)
        logo.pack(side="left")
        tk.Label(logo, text="◈", font=("Segoe UI", 16, "bold"),
                 bg=ACCENT, fg=HDR_TEXT).pack()

        tk.Label(tb, text="SMART ATTENDANCE SYSTEM",
                 font=("Segoe UI", 14, "bold"),
                 bg=HDR_BG, fg=HDR_TEXT).pack(side="left", padx=16)

        self.clock_lbl = tk.Label(tb, text="", font=FONT_SMALL,
                                   bg=HDR_BG, fg="#94A3B8")
        self.clock_lbl.pack(side="left", padx=10)

        ctrl = tk.Frame(tb, bg=HDR_BG)
        ctrl.pack(side="right")

        self.enrolled_lbl = tk.Label(ctrl, text="", font=FONT_SMALL,
                                      bg=HDR_BG, fg="#94A3B8", padx=12)
        self.enrolled_lbl.pack(side="left")
        self._update_enrolled_badge()

    def _update_enrolled_badge(self):
        n = len(self.students)
        self.enrolled_lbl.config(
            text=f"  {n} student{'s' if n != 1 else ''} enrolled  ",
            fg=ACCENT2 if n else WARN)

    # ── Left panel: today's list ──────────────────────────────────────────
    def _build_left(self, parent):
        lp = tk.Frame(parent, bg=BG2, width=220,
                      highlightbackground=BORDER, highlightthickness=1)
        lp.pack(side="left", fill="y", padx=(10, 4), pady=10)
        lp.pack_propagate(False)

        hdr = tk.Frame(lp, bg=NAVY, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="TODAY'S ATTENDANCE",
                 font=FONT_STEP, bg=NAVY, fg=HDR_TEXT).pack()
        self.count_lbl = tk.Label(hdr, text="0 present",
                                   font=FONT_SMALL, bg=NAVY, fg="#93C5FD")
        self.count_lbl.pack()

        canvas = tk.Canvas(lp, bg=BG2, highlightthickness=0)
        sb     = tk.Scrollbar(lp, orient="vertical", command=canvas.yview,
                              bg=BG3, troughcolor=BG3)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._list_frame  = tk.Frame(canvas, bg=BG2)
        self._list_window = canvas.create_window((0, 0),
                                                  window=self._list_frame, anchor="nw")
        self._list_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(self._list_window, width=e.width))

    def add_attendee(self, student, frame_bgr):
        idx  = len(self.attendees_today) + 1
        card = tk.Frame(self._list_frame, bg=BG2, padx=8, pady=8,
                        highlightbackground=ACCENT2, highlightthickness=1)
        card.pack(fill="x", padx=6, pady=3)

        if frame_bgr is not None:
            try:
                img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                img   = img.resize((40, 40))
                photo = ImageTk.PhotoImage(img)
                lbl   = tk.Label(card, image=photo, bg=BG2)
                lbl.image = photo
                lbl.pack(side="left", padx=(0, 8))
            except Exception:
                pass

        tk.Label(card, text=str(idx), font=("Segoe UI", 10, "bold"),
                 bg=ACCENT2, fg=BTN_FG, width=3, pady=2).pack(side="left")

        inf = tk.Frame(card, bg=BG2)
        inf.pack(side="left", padx=8)
        tk.Label(inf, text=student["name"],
                 font=("Segoe UI", 10, "bold"), bg=BG2, fg=TEXT).pack(anchor="w")
        tk.Label(inf, text=student["roll"],
                 font=FONT_SMALL, bg=BG2, fg=MUTED).pack(anchor="w")
        tk.Label(inf, text=datetime.now().strftime("%H:%M:%S"),
                 font=FONT_SMALL, bg=BG2, fg=ACCENT2).pack(anchor="w")

        self.attendees_today.append(student)
        self.count_lbl.config(text=f"{len(self.attendees_today)} present")

    # ── Centre panel: camera + steps ────────────────────────────────────
    def _build_centre(self, parent):
        cp = tk.Frame(parent, bg=BG)
        cp.pack(side="left", fill="both", expand=True, padx=4, pady=10)

        cam_card = tk.Frame(cp, bg=BG2,
                            highlightbackground=BORDER, highlightthickness=1)
        cam_card.pack()
        self.cam_lbl = tk.Label(cam_card, bg="#E2E8F0")
        self.cam_lbl.pack(padx=3, pady=3)

        sb = tk.Frame(cp, bg=BG3,
                      highlightbackground=BORDER, highlightthickness=1)
        sb.pack(fill="x", pady=(6, 4))
        self.status_lbl = tk.Label(
            sb, text="Press ▶ START ATTENDANCE to begin",
            font=FONT_BODY, bg=BG3, fg=MUTED, pady=7, padx=12)
        self.status_lbl.pack(side="left")

        # GPS label
        self.gps_lbl = tk.Label(
            sb, text="📍 ...", font=FONT_SMALL, bg=BG3, fg=MUTED, pady=7, padx=8)
        self.gps_lbl.pack(side="right")
        self._refresh_gps_label()

        # Step indicators
        sf = tk.Frame(cp, bg=BG)
        sf.pack(fill="x", pady=6)
        self._step_ws = []
        for i, (key, label, desc) in enumerate(STEP_DEFS):
            box = tk.Frame(sf, bg=BG2, padx=10, pady=10,
                           highlightbackground=STEP_TODO, highlightthickness=2)
            box.grid(row=0, column=i, padx=4, sticky="nsew")
            sf.columnconfigure(i, weight=1)
            num = tk.Label(box, text=str(i + 1),
                           font=("Segoe UI", 18, "bold"),
                           bg=STEP_TODO, fg=BTN_FG, width=2, pady=2)
            num.pack()
            tk.Label(box, text=label, font=FONT_STEP, bg=BG2, fg=MUTED).pack(pady=(4, 0))
            tk.Label(box, text=desc, font=FONT_SMALL, bg=BG2, fg=MUTED,
                     justify="center").pack()
            self._step_ws.append({"box": box, "num": num})

        # Action row
        ar = tk.Frame(cp, bg=BG)
        ar.pack(pady=8)

        self._btn(ar, "↺  Retry", self._retry,
                  bg=BG3, fg=TEXT, padx=14, pady=8).pack(side="left", padx=4)

        self._session_btn = tk.Button(
            ar, text="▶  START ATTENDANCE",
            font=FONT_BTN, bg=ACCENT2, fg=BTN_FG,
            relief="flat", cursor="hand2", padx=18, pady=8,
            command=self._toggle_session)
        self._session_btn.pack(side="left", padx=4)
        self._session_btn.bind("<Enter>",
            lambda e: self._session_btn.config(bg=NAVY))
        self._session_btn.bind("<Leave>",
            lambda e: self._session_btn.config(
                bg=ACCENT2 if not self._session_active else WARN))

        self._btn(ar, "➕  Enroll New", self._open_enrollment,
                  bg=ACCENT, padx=14, pady=8).pack(side="left", padx=4)

    def _btn(self, parent, text, cmd, bg=None, fg=None, **kw):
        b = bg or ACCENT
        f = fg or BTN_FG
        btn = tk.Button(parent, text=text, command=cmd, bg=b, fg=f,
                        font=FONT_BTN, relief="flat", cursor="hand2",
                        activebackground=NAVY, activeforeground=BTN_FG, bd=0, **kw)
        btn.bind("<Enter>", lambda e: btn.config(bg=NAVY))
        btn.bind("<Leave>", lambda e: btn.config(bg=b))
        return btn

    # ── Camera display ────────────────────────────────────────────────────
    def update_frame(self, frame_bgr, result):
        disp  = self._draw_frame(frame_bgr, result)
        rgb   = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.cam_lbl.config(image=photo)
        self.cam_lbl.image = photo

    def update_frame_idle(self, frame_bgr):
        """Dimmed live feed shown when session is stopped."""
        disp = cv2.resize(frame_bgr, (CAM_W, CAM_H))
        h, w = disp.shape[:2]
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.45, disp, 0.55, 0, disp)
        for msg, y_off, scale, thick in [
            ("ATTENDANCE SESSION INACTIVE", -22, 0.75, 2),
            ("Press  START ATTENDANCE  to begin", +18, 0.52, 1),
        ]:
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(disp, msg, ((w - tw) // 2, h // 2 + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), thick)
        rgb   = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.cam_lbl.config(image=photo)
        self.cam_lbl.image = photo

    def _draw_frame(self, frame_bgr, result):
        disp  = cv2.resize(frame_bgr, (CAM_W, CAM_H))
        scale = CAM_W / frame_bgr.shape[1]

        # Face box
        if result.face is not None:
            x, y, w, h = result.face
            x2, y2 = int(x * scale), int(y * scale)
            w2, h2 = int(w * scale), int(h * scale)
            col = (63, 185, 80) if result.step != S_FACE else (88, 166, 255)
            cv2.rectangle(disp, (x2, y2), (x2 + w2, y2 + h2), col, 2)

        # Hand overlay
        if result.hand_kpts:
            info    = result.hand_kpts
            bbox    = info.get("bbox")
            shoulder= info.get("shoulder")
            is_r    = info.get("is_raised", False)
            if bbox:
                bx, by, bw, bh = bbox
                scaled = {
                    "bbox": (int(bx*scale), int(by*scale),
                             int(bw*scale), int(bh*scale)),
                    "shoulder": (int(shoulder[0]*scale), int(shoulder[1]*scale))
                                if shoulder else None,
                }
                disp = draw_hand_overlay(disp, None, scaled, is_r, result.gate_ok)

        # Overlay text banner
        if result.overlay_text:
            overlay = disp.copy()
            cv2.rectangle(overlay, (0, 0), (CAM_W, 48), (13, 17, 23), -1)
            cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)
            col = (63, 185, 80) if result.gate_ok else (88, 166, 255)
            cv2.putText(disp, result.overlay_text, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return disp

    # ── Step indicators ───────────────────────────────────────────────────
    def update_step(self, step, msg, tone):
        col = {"green": ACCENT2, "warn": WARN, "accent": ACCENT, "muted": MUTED}.get(tone, MUTED)
        if msg:
            self.status_lbl.config(text=msg, fg=col)
        self._cur_step = step
        self.refresh_steps(step)

    def refresh_steps(self, current_step):
        idx_map = {S_FACE: 0, S_HAND: 1, "hand_prompt": 1,
                   S_LIVENESS: 2, S_VOICE: 3, S_DONE: 4}
        cur_i   = idx_map.get(current_step, 0)
        for i, sw in enumerate(self._step_ws):
            if i < cur_i:
                sw["box"].config(highlightbackground=STEP_DONE)
                sw["num"].config(bg=STEP_DONE, fg=BTN_FG, text="✓")
            elif i == cur_i:
                sw["box"].config(highlightbackground=STEP_NOW)
                sw["num"].config(bg=STEP_NOW, fg=BTN_FG, text=str(i + 1))
            else:
                sw["box"].config(highlightbackground=STEP_TODO)
                sw["num"].config(bg=STEP_TODO, fg=BTN_FG, text=str(i + 1))

    # ── Right panel: result ───────────────────────────────────────────────
    def _build_right(self, parent):
        rp = tk.Frame(parent, bg=BG2, width=240,
                      highlightbackground=BORDER, highlightthickness=1)
        rp.pack(side="right", fill="y", padx=(4, 10), pady=10)
        rp.pack_propagate(False)

        hdr = tk.Frame(rp, bg=NAVY, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="VERIFICATION RESULT",
                 font=FONT_STEP, bg=NAVY, fg=HDR_TEXT).pack()

        self.snap_lbl = tk.Label(rp, bg=BG3, width=220, height=160,
                                  text="No capture yet", font=FONT_SMALL, fg=MUTED)
        self.snap_lbl.pack(padx=10, pady=10)

        inf = tk.Frame(rp, bg=BG2, padx=10)
        inf.pack(fill="x")
        self._info = {}
        for key, label in [("name", "Name"), ("roll", "Roll No"),
                            ("time", "Time"), ("date", "Date"),
                            ("gps", "Location"), ("status", "Status")]:
            r = tk.Frame(inf, bg=BG3, pady=4, padx=8,
                         highlightbackground=BORDER, highlightthickness=1)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=f"{label}:", font=FONT_STEP, bg=BG3, fg=MUTED,
                     width=8, anchor="w").pack(side="left")
            v = tk.Label(r, text="—", font=FONT_SMALL, bg=BG3, fg=TEXT,
                         anchor="w", wraplength=130)
            v.pack(side="left")
            self._info[key] = v

        self.banner_lbl = tk.Label(rp, text="WAITING",
                                    font=("Segoe UI", 13, "bold"),
                                    bg=BG2, fg=MUTED, pady=10)
        self.banner_lbl.pack()

    def show_marked(self, student, photo_path, dt, voice_skipped=False):
        _, _, address = get_location()
        self._info["name"].config(text=student["name"], fg=TEXT)
        self._info["roll"].config(text=student["roll"], fg=TEXT)
        self._info["time"].config(text=dt.strftime("%H:%M:%S"), fg=ACCENT2)
        self._info["date"].config(text=dt.strftime("%Y-%m-%d"), fg=TEXT)
        self._info["gps"].config(text=address or "—", fg=MUTED)
        status = "Present ✓" + (" (flagged)" if voice_skipped else "")
        self._info["status"].config(text=status, fg=ACCENT2)
        self.banner_lbl.config(text="✓  MARKED", fg=BTN_FG, bg=ACCENT2)
        self.status_lbl.config(
            text=f"✅  Marked — {student['name']}  {dt.strftime('%H:%M:%S')}", fg=ACCENT2)

        if photo_path and os.path.isfile(photo_path):
            try:
                img   = Image.open(photo_path).resize((220, 160))
                draw  = ImageDraw.Draw(img)
                draw.rectangle([0, 0, 219, 159], outline=(22, 163, 74), width=3)
                photo = ImageTk.PhotoImage(img)
                self.snap_lbl.config(image=photo, text="")
                self.snap_lbl.image = photo
            except Exception:
                pass

    def show_already_marked(self, student):
        self.banner_lbl.config(text="⚠  ALREADY MARKED", fg=BTN_FG, bg=WARN)
        self.status_lbl.config(
            text=f"⚠  {student['name']} already marked today", fg=WARN)
        self._info["name"].config(text=student["name"], fg=TEXT)
        self._info["roll"].config(text=student["roll"], fg=TEXT)
        self._info["status"].config(text="Already present ✓", fg=WARN)

    def reset_right(self):
        self.banner_lbl.config(text="WAITING", fg=MUTED, bg=BG2)
        self.snap_lbl.config(image="", text="No capture yet", bg=BG3)
        for k in self._info:
            self._info[k].config(text="—", fg=TEXT)

    # ── Session controls ──────────────────────────────────────────────────
    def _toggle_session(self):
        if not self._session_active:
            if self._on_session_start_cb:
                self._on_session_start_cb()
        else:
            if self._on_session_stop_cb:
                self._on_session_stop_cb()

    def set_session_ui(self, active: bool):
        self._session_active = active
        if active:
            self._session_btn.config(text="■  STOP SESSION", bg=WARN)
        else:
            self._session_btn.config(text="▶  START ATTENDANCE", bg=ACCENT2)
            self.reset_right()
            self.refresh_steps(S_FACE)

    # ── GPS label (refreshed periodically) ───────────────────────────────
    def _refresh_gps_label(self):
        _, _, address = get_location()
        if address and address != "—":
            self.gps_lbl.config(text=f"📍 {address}")
        else:
            self.gps_lbl.config(text="📍 Fetching location…")
        self.after(5000, self._refresh_gps_label)

    # ── Misc ──────────────────────────────────────────────────────────────
    def _retry(self):
        self.reset_right()
        self.refresh_steps(S_FACE)
        self.status_lbl.config(text="Ready — stand in front of the camera", fg=MUTED)
        self.on_pipeline_reset()

    def _open_enrollment(self):
        from enrollment_app import EnrollmentApp
        from enroll_config import WIN_WIDTH, WIN_HEIGHT
        win = tk.Toplevel(self)
        win.title("Enrollment")
        win.configure(bg="#F0F4F8")
        sx, sy = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}"
                     f"+{(sx-WIN_WIDTH)//2}+{(sy-WIN_HEIGHT)//2}")
        EnrollmentApp(win)
        win.protocol("WM_DELETE_WINDOW",
            lambda: (win.destroy(), self.after(600, self._on_enrollment_closed_cb)
            if self._on_enrollment_closed_cb else None
        ))

    def _reload_students(self):
        from attend_detectors import load_students
        self.students = load_students()
        self._update_enrolled_badge()
        self.status_lbl.config(
            text=f"Reloaded — {len(self.students)} students found", fg=ACCENT)

    def _tick_clock(self):
        self.clock_lbl.config(
            text=datetime.now().strftime("%A  %d %b %Y   %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def _on_close(self):
        self.destroy()
