"""
attend_detectors.py  —  detection functions
Works with both old (<=0.9) and new (>=0.10) MediaPipe.
Falls back to OpenCV-only when MediaPipe is absent / incompatible.
"""

import cv2
import numpy as np
import os
import json
import csv
import time
import threading
from datetime import datetime
from attend_config import (
    STUDENTS_DIR, ATTENDANCE_DIR, FACE_HIST_BINS, FACE_MATCH_THRESH,
    SAMPLE_RATE, VOICE_MATCH_THRESH, GPS_ENABLED, GPS_TIMEOUT_S,
)

# ── Haar cascades ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

def _log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MediaPipe — graceful multi-version support                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MEDIAPIPE_AVAILABLE = False
_mp_hands_detector  = None
_mp_face_mesh       = None

try:
    import mediapipe as mp

    # ── Legacy API (mediapipe ≤ 0.9) ─────────────────────────────────────
    if hasattr(mp, "solutions"):
        _mp_hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        MEDIAPIPE_AVAILABLE = True
        _log("[ok] MediaPipe legacy API initialised")

    # ── New Tasks API (mediapipe ≥ 0.10) — needs .task model files ───────
    else:
        # Model files must live next to this script.
        # Download once:
        #   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
        #   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        _hand_model = os.path.join(_THIS_DIR, "hand_landmarker.task")
        _face_model = os.path.join(_THIS_DIR, "face_landmarker.task")

        if os.path.isfile(_hand_model) and os.path.isfile(_face_model):
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            _hand_opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_hand_model),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            _mp_hands_detector = vision.HandLandmarker.create_from_options(_hand_opts)

            _face_opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_face_model),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            _mp_face_mesh = vision.FaceLandmarker.create_from_options(_face_opts)

            MEDIAPIPE_AVAILABLE = True
            _log("[ok] MediaPipe Tasks API initialised (new 0.10+ style)")
        else:
            _log("[warn] MediaPipe 0.10+ detected but model files missing.")
            _log("  Falling back to OpenCV-only detection (works fine).")
            _log("  To enable MediaPipe: download hand_landmarker.task and")
            _log("  face_landmarker.task into the same folder as this script.")

except ImportError:
    _log("[info] MediaPipe not installed — using OpenCV-only detection (works fine)")
except Exception as e:
    _log(f"[warn] MediaPipe init failed ({e}) — using OpenCV-only detection")


# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_OK = False
sd       = None
try:
    import sounddevice as _sd
    # Quick test — will raise if PortAudio not found
    _sd.query_devices()
    sd = _sd
    AUDIO_OK = True
    _log("[ok] Audio (sounddevice) ready")
except Exception as e:
    _log(f"[info] Audio not available: {e}  — voice step will be skipped")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GPS / Location                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_gps_cache = {"lat": None, "lon": None, "address": None, "fetched": False}

def _fetch_gps_once():
    """Fetch location via ip-api.com (free, no key needed).
    Runs in background thread so it never blocks the camera."""
    if _gps_cache["fetched"]:
        return
    _gps_cache["fetched"] = True
    try:
        import urllib.request, json as _json
        url  = "http://ip-api.com/json/?fields=lat,lon,city,regionName,country"
        req  = urllib.request.urlopen(url, timeout=GPS_TIMEOUT_S)
        data = _json.loads(req.read().decode())
        _gps_cache["lat"]     = data.get("lat")
        _gps_cache["lon"]     = data.get("lon")
        _gps_cache["address"] = (
            f"{data.get('city','')}, {data.get('regionName','')}, {data.get('country','')}"
        )
        _log(f"[ok] Location: {_gps_cache['address']}  "
             f"({_gps_cache['lat']}, {_gps_cache['lon']})")
    except Exception as e:
        _gps_cache["address"] = "GPS unavailable"
        _log(f"[info] GPS lookup failed: {e}")

if GPS_ENABLED:
    threading.Thread(target=_fetch_gps_once, daemon=True).start()


def get_location():
    """Return (lat, lon, address_str) — values may be None if not yet fetched."""
    return _gps_cache["lat"], _gps_cache["lon"], _gps_cache.get("address", "—")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Camera helper                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def open_camera(preferred_index=0):
    """Try preferred_index first, then scan 0-4 for a working camera."""
    cap = cv2.VideoCapture(preferred_index)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            _log(f"[ok] Camera opened at index {preferred_index}")
            return cap
    cap.release()
    for idx in range(5):
        if idx == preferred_index:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                _log(f"[ok] Camera opened at index {idx}")
                return cap
        cap.release()
    return None   # caller must handle None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Face                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def detect_face(frame_bgr):
    """Return (x,y,w,h) of the largest face or None."""
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if not len(faces):
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def face_histogram(img_bgr):
    """Hue + saturation + edge-magnitude histogram feature vector."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    img  = cv2.resize(img_bgr, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # try to crop to face area for better matching
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = img[y:y + h, x:x + w]
        if crop.size:
            img  = cv2.resize(crop, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch = cv2.calcHist([hsv], [0], None, [FACE_HIST_BINS], [0, 180]).flatten()
    s_ch = cv2.calcHist([hsv], [1], None, [FACE_HIST_BINS], [0, 256]).flatten()
    gx   = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)
    gy   = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
    mag  = np.sqrt(gx ** 2 + gy ** 2).flatten()
    t_ch = np.histogram(mag, bins=64, range=(0, 300))[0].astype(np.float32)
    feat = np.concatenate([h_ch, s_ch, t_ch])
    n    = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


def match_face(face_crop_bgr, students):
    """Return (best_student, similarity) or (None, 0)."""
    hist = face_histogram(face_crop_bgr)
    if hist is None:
        return None, 0.0
    best_s, best_sim = None, 0.0
    for s in students:
        for sh in s["face_hists"]:
            n1, n2 = np.linalg.norm(hist), np.linalg.norm(sh)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            sim = float(np.dot(hist, sh) / (n1 * n2))
            if sim > best_sim:
                best_sim, best_s = sim, s
    return (best_s, best_sim) if best_sim >= FACE_MATCH_THRESH else (None, best_sim)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Hand Detection                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def detect_raised_hand(frame_bgr, face_box=None):
    """
    Returns (is_raised, landmarks_or_None, bbox_or_None, wrist_y, shoulder).
    Uses MediaPipe if available, else skin-colour fallback.
    """
    if MEDIAPIPE_AVAILABLE and _mp_hands_detector is not None:
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── Legacy (≤0.9) ─────────────────────────────────────────────
            if hasattr(mp, "solutions"):
                results = _mp_hands_detector.process(rgb)
                if not results.multi_hand_landmarks:
                    return False, None, None, None, None
                hand = results.multi_hand_landmarks[0]
                h, w = frame_bgr.shape[:2]
                lms  = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
                return _hand_from_lms(lms, frame_bgr, face_box)

            # ── New Tasks (≥0.10) ──────────────────────────────────────────
            else:
                from mediapipe import Image as MpImage, ImageFormat
                mp_img  = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                results = _mp_hands_detector.detect(mp_img)
                if not results.hand_landmarks:
                    return False, None, None, None, None
                h, w = frame_bgr.shape[:2]
                lms = [(int(lm.x * w), int(lm.y * h))
                       for lm in results.hand_landmarks[0]]
                return _hand_from_lms(lms, frame_bgr, face_box)

        except Exception as e:
            _log(f"[warn] Hand MP error: {e}")

    return _detect_hand_skin_fallback(frame_bgr, face_box)


def _hand_from_lms(lms, frame_bgr, face_box):
    h, w = frame_bgr.shape[:2]
    xs   = [p[0] for p in lms]
    ys   = [p[1] for p in lms]
    bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    wrist_y = lms[0][1]

    face = face_box or detect_face(frame_bgr)
    if face:
        fx, fy, fw, fh = face
        shoulder_x   = fx + fw // 2
        shoulder_y   = fy + fh + 40
    else:
        shoulder_x = w // 2
        shoulder_y = h // 3

    is_raised = wrist_y < shoulder_y - 20
    return is_raised, lms, bbox, wrist_y, (shoulder_x, shoulder_y)


def _detect_hand_skin_fallback(frame_bgr, face_box=None):
    """Skin-colour blob fallback — no model files needed."""
    try:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Broad skin range (works for most skin tones under typical lighting)
        mask1 = cv2.inRange(hsv, np.array([0,  20,  70]), np.array([20, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180,255,255]))
        mask  = cv2.bitwise_or(mask1, mask2)

        fh, fw = frame_bgr.shape[:2]
        if face_box is not None:
            fx, fy, face_w, face_h = face_box
            pad = 30
            mask[max(0, fy-pad):min(fh, fy+face_h+pad),
                 max(0, fx-pad):min(fw, fx+face_w+pad)] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False, None, None, None, None

        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < 2000:
            return False, None, None, None, None

        bbox    = cv2.boundingRect(largest)
        bx, by, bw, bh = bbox
        hand_cx = bx + bw // 2
        hand_cy = by + bh // 2

        if face_box is not None:
            fx, fy, face_w, face_h = face_box
            shoulder_x = fx + face_w // 2
            shoulder_y = fy + face_h + 40
        else:
            shoulder_x = fw // 2
            shoulder_y = fh // 3

        is_raised = hand_cy < shoulder_y - 20
        return is_raised, None, bbox, hand_cy, (shoulder_x, shoulder_y)

    except Exception as e:
        _log(f"[warn] Skin hand detect error: {e}")
        return False, None, None, None, None


def draw_hand_overlay(frame, landmarks, bbox_info, is_raised, confirmed):
    """Draw hand bounding box and shoulder-to-hand arm line."""
    if bbox_info is None:
        return frame

    shoulder = None
    if isinstance(bbox_info, dict):
        shoulder = bbox_info.get("shoulder")
        bbox_raw = bbox_info.get("bbox")
    else:
        bbox_raw = bbox_info

    if bbox_raw is None:
        return frame

    bx, by, bw, bh = bbox_raw
    col = (0, 230, 80) if confirmed else (0, 200, 255) if is_raised else (100, 100, 100)

    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), col, 2)
    cx = bx + bw // 2
    cy = by + bh // 2

    if shoulder:
        sx, sy = shoulder
        cv2.circle(frame, (sx, sy), 10, (255, 120, 0), -1)
        cv2.line(frame, (sx, sy), (cx, cy), col, 3)
        for t in np.linspace(0.2, 0.8, 5):
            px = int(sx * (1 - t) + cx * t)
            py = int(sy * (1 - t) + cy * t)
            cv2.circle(frame, (px, py), 5, col, -1)

    label = "✓ CONFIRMED" if confirmed else "HAND RAISED" if is_raised else "RAISE HAND"
    cv2.putText(frame, label, (bx, max(by - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return frame


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Liveness  (blink + nod)                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def detect_liveness(frame_bgr, face_box=None, prev_state=None):
    """
    Returns (blink_ok, nod_ok, ear, nod_magnitude, new_state).
    Tries MediaPipe FaceMesh → OpenCV eye cascade fallback.
    """
    if MEDIAPIPE_AVAILABLE and _mp_face_mesh is not None and face_box is not None:
        try:
            import mediapipe as mp
            x, y, w, h = face_box
            roi = frame_bgr[y:y+h, x:x+w]
            if roi.size == 0:
                raise ValueError("empty ROI")
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            landmarks = None
            # ── Legacy ────────────────────────────────────────────────────
            if hasattr(mp, "solutions"):
                res = _mp_face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    landmarks = res.multi_face_landmarks[0].landmark
                    hw, ww = roi.shape[:2]
                    lms = [(lm.x, lm.y) for lm in landmarks]
            # ── New Tasks ─────────────────────────────────────────────────
            else:
                from mediapipe import Image as MpImage, ImageFormat
                mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
                res    = _mp_face_mesh.detect(mp_img)
                if res.face_landmarks:
                    lms = [(lm.x, lm.y) for lm in res.face_landmarks[0]]
                    landmarks = True   # just a truthy marker

            if landmarks is not None and lms:
                return _liveness_from_landmarks(lms, face_box, prev_state)
        except Exception as e:
            _log(f"[warn] Liveness MP error: {e}")

    return _detect_liveness_opencv(frame_bgr, face_box, prev_state)


def _liveness_from_landmarks(lms, face_box, prev_state):
    """Compute EAR blink + nod from (x,y) normalised landmarks."""
    if prev_state is None:
        prev_state = {
            "nose_y_history": [], "blink_counter": 0,
            "blink_confirmed": False, "last_blink_t": 0,
        }

    # MediaPipe face mesh EAR indices (standard 468-point model)
    # Left eye:  33,159,145,133,153,144   Right eye: 362,386,374,263,380,373
    def _ear(idx_list):
        pts = [lms[i] for i in idx_list]
        v1 = np.linalg.norm(np.subtract(pts[1], pts[5]))
        v2 = np.linalg.norm(np.subtract(pts[2], pts[4]))
        h  = np.linalg.norm(np.subtract(pts[0], pts[3]))
        return (v1 + v2) / (2.0 * h + 1e-6)

    try:
        left_ear  = _ear([33, 159, 145, 133, 153, 144])
        right_ear = _ear([362, 386, 374, 263, 380, 373])
        ear       = (left_ear + right_ear) / 2.0
    except IndexError:
        ear = 0.3   # assume open

    BLINK_THRESH = 0.22
    is_blinking  = ear < BLINK_THRESH

    if is_blinking:
        prev_state["blink_counter"] += 1
    else:
        if prev_state["blink_counter"] >= 2:
            prev_state["blink_confirmed"] = True
            prev_state["last_blink_t"] = time.time()
        prev_state["blink_counter"] = 0

    # Nod: track nose tip (landmark 1 in 468-pt model, index 1)
    try:
        nose_y = lms[1][1]
    except IndexError:
        nose_y = 0.5

    prev_state["nose_y_history"].append(nose_y)
    if len(prev_state["nose_y_history"]) > 40:
        prev_state["nose_y_history"].pop(0)

    nod_mag = 0
    nod_ok  = False
    if len(prev_state["nose_y_history"]) >= 15:
        nod_mag = (max(prev_state["nose_y_history"]) -
                   min(prev_state["nose_y_history"])) * (face_box[3] if face_box else 200)
        nod_ok  = nod_mag > 20

    blink_ok = prev_state["blink_confirmed"] or (
        time.time() - prev_state["last_blink_t"] < 1.0)

    return blink_ok, nod_ok, ear, nod_mag, prev_state


def _detect_liveness_opencv(frame_bgr, face_box=None, prev_state=None):
    """OpenCV-only blink/nod — no model files needed."""
    if face_box is None:
        return False, False, 0, 0, prev_state

    if prev_state is None:
        prev_state = {
            "face_y_history": [], "open_eye_seen": False,
            "closed_eye_frames": 0, "blink_confirmed": False,
        }

    x, y, w, h = face_box
    roi  = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return False, False, 0, 0, prev_state

    gray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upper     = gray[:max(1, h//2), :]
    eyes      = eye_cascade.detectMultiScale(
        upper, scaleFactor=1.1, minNeighbors=4,
        minSize=(max(12, w//12), max(8, h//18)))
    eye_count = len(eyes)

    if eye_count >= 1:
        prev_state["open_eye_seen"] = True
        if 0 < prev_state["closed_eye_frames"] <= 8:
            prev_state["blink_confirmed"] = True
        prev_state["closed_eye_frames"] = 0
    elif prev_state["open_eye_seen"]:
        prev_state["closed_eye_frames"] += 1
        if prev_state["closed_eye_frames"] >= 2:
            prev_state["blink_confirmed"] = True

    cy = y + h / 2.0
    prev_state["face_y_history"].append(cy)
    if len(prev_state["face_y_history"]) > 35:
        prev_state["face_y_history"].pop(0)

    nod_mag = 0
    nod_ok  = False
    if len(prev_state["face_y_history"]) > 10:
        nod_mag = (max(prev_state["face_y_history"]) -
                   min(prev_state["face_y_history"]))
        nod_ok  = nod_mag > 18

    ear = min(eye_count / 2.0, 1.0)
    return prev_state["blink_confirmed"], nod_ok, ear, nod_mag, prev_state


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Voice                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def record_voice(seconds=3):
    if not AUDIO_OK or sd is None:
        return None
    try:
        _log(f"\n[recording] {seconds}s — speak now!")
        rec = sd.rec(int(seconds * SAMPLE_RATE),
                     samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        if rec is None or len(rec) == 0:
            return None
        if np.max(np.abs(rec)) < 0.01:
            _log("[warn] Recording too quiet")
            return None
        _log("[ok] Voice recorded")
        return rec.flatten()
    except Exception as e:
        _log(f"[error] Record failed: {e}")
        return None


def _extract_voice_features(audio, n_bands=26):
    if audio is None or len(audio) == 0:
        return np.zeros(n_bands, dtype=np.float32)
    try:
        audio = audio.flatten().astype(np.float32)
        mx    = np.max(np.abs(audio))
        if mx > 0:
            audio /= mx
        frame_len = int(SAMPLE_RATE * 0.025)
        step      = int(SAMPLE_RATE * 0.010)
        if len(audio) < frame_len:
            return np.zeros(n_bands, dtype=np.float32)
        min_f, max_f = 80, 8000
        edges = np.logspace(np.log10(min_f), np.log10(max_f), n_bands + 1)
        from scipy.fft import rfft, rfftfreq
        feats = []
        for start in range(0, len(audio) - frame_len, step):
            frame    = audio[start:start + frame_len] * np.hanning(frame_len)
            spectrum = np.abs(rfft(frame))
            freqs    = rfftfreq(frame_len, 1 / SAMPLE_RATE)
            bands = [np.sum(spectrum[(freqs >= edges[i]) & (freqs < edges[i+1])])
                     for i in range(n_bands)]
            total = sum(bands) + 1e-6
            feats.append([b / total for b in bands])
        if not feats:
            return np.zeros(n_bands, dtype=np.float32)
        avg  = np.mean(feats, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 1e-6 else avg
    except Exception as e:
        _log(f"[warn] Feature extraction failed: {e}")
        return np.zeros(n_bands, dtype=np.float32)


def match_voice(audio, student):
    if student is None or student.get("voice_feat") is None:
        return True, 1.0          # no voice enrolled → skip
    if audio is None:
        return False, 0.0
    stored = student["voice_feat"]
    feat   = _extract_voice_features(audio, n_bands=stored.shape[0])
    n1, n2 = np.linalg.norm(feat), np.linalg.norm(stored)
    if n1 < 1e-6 or n2 < 1e-6:
        return False, 0.0
    sim = float(np.dot(feat, stored) / (n1 * n2))
    return sim >= VOICE_MATCH_THRESH, sim


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Students & Logging                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def load_students():
    students = []
    if not os.path.isdir(STUDENTS_DIR):
        return students
    for entry in os.listdir(STUDENTS_DIR):
        folder = os.path.join(STUDENTS_DIR, entry)
        meta_f = os.path.join(folder, "metadata.json")
        if not os.path.isdir(folder) or not os.path.isfile(meta_f):
            continue
        try:
            with open(meta_f) as f:
                meta = json.load(f)
            hists = []
            for fname in meta.get("photos", []):
                img_path = os.path.join(folder, fname)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        h = face_histogram(img)
                        if h is not None:
                            hists.append(h)
            feat_path = os.path.join(folder, "voice_features.npy")
            voice_feat = None
            if os.path.isfile(feat_path):
                vf = np.load(feat_path).astype(np.float32)
                if vf.ndim == 1 and np.all(np.isfinite(vf)) and np.linalg.norm(vf) > 1e-6:
                    voice_feat = vf
                else:
                    _log(f"[warn] Bad voice features for {entry} — skipping voice")
            students.append({
                "name": meta["name"],
                "roll": meta["roll_number"],
                "folder": folder,
                "face_hists": hists,
                "voice_feat": voice_feat,
            })
        except Exception as e:
            _log(f"[error] Loading student {entry}: {e}")
    _log(f"[ok] Loaded {len(students)} student(s)")
    return students


def log_attendance(student, photo_path="", voice_skipped=False):
    """Write one CSV row for today + include GPS."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today    = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    is_new   = not os.path.isfile(csv_path)
    now      = datetime.now()
    lat, lon, address = get_location()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["Roll", "Name", "Date", "Time",
                         "Lat", "Lon", "Location", "Photo", "VoiceSkipped"])
        w.writerow([
            student["roll"], student["name"],
            now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
            lat or "", lon or "", address or "",
            photo_path, "yes" if voice_skipped else "no",
        ])
    _log(f"[ok] Logged: {student['name']} ({student['roll']}) @ {address}")
