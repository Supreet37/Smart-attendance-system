# Smart Attendance System — Setup & Troubleshooting

## Quick Start
```
pip install -r requirements.txt
python enrollment_app.py   # enroll students first
python attendance_app.py   # then take attendance
```

---

## Camera won't open?
Edit `attend_config.py` and change:
```python
CAMERA_INDEX = 0   # try 1, 2, 3 if 0 doesn't work
```
On **Linux** also run:  `sudo usermod -aG video $USER`  then log out & in.

---

## MediaPipe (hand / face mesh) not working?
The system works perfectly **without** MediaPipe using OpenCV fallbacks.
If you want MediaPipe and have version ≥ 0.10:

1. Download the two model files into the same folder as `attend_detectors.py`:
   - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   - https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

---

## Audio / voice not working?
- **Windows**: usually works out of the box.
- **Linux**: `sudo apt install portaudio19-dev libsndfile1` then `pip install sounddevice`.
- **macOS**: `brew install portaudio` then `pip install sounddevice`.
- Voice step is **automatically skipped** if audio is unavailable — attendance is still marked.

---

## GPS Location
- Uses free IP geolocation (ip-api.com) — no API key needed.
- Runs in background; doesn't block the camera.
- Recorded in the CSV: Lat, Lon, City/Region/Country columns.
- Disable by setting `GPS_ENABLED = False` in `attend_config.py`.

---

## CSV Output
`attendance/attendance_YYYY-MM-DD.csv`

Columns: Roll, Name, Date, Time, Lat, Lon, Location, Photo, VoiceSkipped

Photos saved to: `attendance/YYYY-MM-DD/<roll>_HHMMSS.jpg`

---

## Files Overview
| File | Purpose |
|---|---|
| `attendance_app.py` | **Main entry point** — run this |
| `attend_config.py` | All settings (camera index, thresholds, etc.) |
| `attend_detectors.py` | Face / hand / liveness / voice / GPS detection |
| `attend_pipeline.py` | 4-step verification state machine |
| `attend_ui.py` | Tkinter UI |
| `enrollment_app.py` | Student enrollment window |
| `enroll_camera.py` | Camera logic for enrollment |
| `enroll_voice.py` | Voice recording for enrollment |
| `enroll_config.py` | Enrollment settings |
