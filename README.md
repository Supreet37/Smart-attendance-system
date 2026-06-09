# 🎓 Smart Attendance System — Installation & User Guide

A secure AI-powered attendance system that combines **face recognition, liveness detection, gesture verification, voice confirmation, and location logging** to prevent proxy attendance and improve reliability.

---

## 🚀 Getting Started

Install the required dependencies and launch the applications:

```bash
pip install -r requirements.txt

# Step 1: Enroll students
python enrollment_app.py

# Step 2: Start attendance verification
python attendance_app.py
```

> Students must be enrolled before attendance can be recorded.

---

## 📷 Camera Troubleshooting

If the webcam does not open, edit `attend_config.py` and try a different camera index:

```python
CAMERA_INDEX = 0
```

Possible values:

```python
CAMERA_INDEX = 1
CAMERA_INDEX = 2
CAMERA_INDEX = 3
```

### Linux Users

Grant camera permissions:

```bash
sudo usermod -aG video $USER
```

Log out and log back in after running the command.

---

## 🤚 MediaPipe Hand & Face Detection

The system includes OpenCV-based fallback detectors and will continue working even if MediaPipe is unavailable.

To enable MediaPipe support (v0.10+), download these model files and place them in the same directory as `attend_detectors.py`:

### Hand Landmarker

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

### Face Landmarker

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

---

## 🎤 Voice Verification Issues

Voice verification adds an extra layer of identity confirmation.

### Windows

Works out of the box in most cases.

### Linux

```bash
sudo apt install portaudio19-dev libsndfile1
pip install sounddevice
```

### macOS

```bash
brew install portaudio
pip install sounddevice
```

> If no microphone is detected, the voice verification step is automatically skipped and attendance can still be recorded successfully.

---

## 📍 GPS & Location Logging

The system can automatically capture approximate attendance location using free IP-based geolocation.

### Features

* No API key required
* Uses `ip-api.com`
* Runs in the background without blocking the camera
* Stores location information alongside attendance records

Recorded fields include:

* Latitude
* Longitude
* City
* Region
* Country

Disable location tracking in `attend_config.py`:

```python
GPS_ENABLED = False
```

---

## 📊 Attendance Records

Attendance logs are automatically generated in CSV format:

```text
attendance/attendance_YYYY-MM-DD.csv
```

### Recorded Information

* Roll Number
* Student Name
* Date
* Time
* Latitude
* Longitude
* Location
* Captured Photo Path
* Voice Verification Status

Captured images are stored in:

```text
attendance/YYYY-MM-DD/<roll>_HHMMSS.jpg
```

---

## 📂 Project Structure

| File                  | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `attendance_app.py`   | Main attendance application entry point                |
| `attend_config.py`    | System configuration and thresholds                    |
| `attend_detectors.py` | Face, hand, liveness, voice, and GPS detection modules |
| `attend_pipeline.py`  | Multi-step attendance verification workflow            |
| `attend_ui.py`        | Tkinter-based user interface                           |
| `enrollment_app.py`   | Student enrollment application                         |
| `enroll_camera.py`    | Enrollment image capture functionality                 |
| `enroll_voice.py`     | Enrollment voice recording module                      |
| `enroll_config.py`    | Enrollment configuration settings                      |

---

## 🔒 Verification Pipeline

The attendance workflow uses multiple verification stages to improve authenticity:

1. Face Recognition
2. Liveness Detection
3. Gesture Verification
4. Voice Confirmation (optional)
5. Location Logging

This layered approach helps reduce spoofing attempts and proxy attendance while maintaining a smooth user experience.

---

## 💡 Notes

* Ensure adequate lighting for reliable face detection.
* Use a working webcam and microphone for the best experience.
* Attendance can still be recorded if optional modules (MediaPipe or microphone) are unavailable.
* All captured data is stored locally on the system.
