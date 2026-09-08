# AI Vision: Hand Tracking & Emotion Detection System

An advanced, real-time computer vision application that detects and tracks hands, counts raised fingers, classifies 8+ hand gestures, and performs real-time facial emotion recognition using OpenCV, MediaPipe, Python 3.11+, and Flask with a futuristic cyber-themed dashboard.

![AI Vision Banner](https://img.shields.io/badge/AI%20Vision-Hand%20Tracking%20%26%20Emotion%20Detection-00f0ff?style=for-the-badge&logo=opencv)

---

## 🌟 Key Features

1. **Real-Time Hand Tracking & 3D Landmarks**:
   - Detects up to 2 hands simultaneously.
   - Extracts 21 3D hand joint landmarks per hand.
   - Accurately identifies `Left` vs `Right` handedness with selfie-mirroring correction.
   - Calculates open/folded state for each finger and displays total raised finger counts.

2. **Real-Time Gesture Recognition**:
   - ✋ **Open Palm**: All 5 fingers extended.
   - ✊ **Fist**: All 5 fingers folded into palm.
   - 👍 **Thumbs Up**: Thumb extended upwards, other fingers folded.
   - 👎 **Thumbs Down**: Thumb extended downwards, other fingers folded.
   - ✌️ **Victory (Peace)**: Index and Middle fingers extended.
   - 👌 **OK**: Thumb and Index fingertips touching, other fingers extended.
   - ☝️ **One Finger**: Index finger extended.
   - 🤟 **Three Fingers**: Index, Middle, and Ring/Thumb extended.

3. **Facial Emotion Detection**:
   - Identifies 7 core human emotions: **Happy**, **Sad**, **Angry**, **Fear**, **Surprise**, **Neutral**, **Disgust**.
   - Computes emotion probability distribution and confidence percentages.
   - Employs frame-skipping inference caching for ultra-high FPS stability.

4. **Futuristic Cyber HUD & Web Dashboard**:
   - Interactive glassmorphic dashboard built with HTML5, CSS3, JavaScript, and Bootstrap 5.
   - Live MJPEG video stream with custom cybernetic bounding boxes, landmark skeletons, and glowing HUD badges.
   - Real-time telemetry monitoring (FPS meter, gesture badges, finger counts, emotion spectrum bar chart).
   - Start / Stop camera hardware controls with clean resource lifecycle management.

---

## 🛠️ Technology Stack

- **Core Engine**: Python 3.11+
- **Computer Vision**: OpenCV (`cv2`), MediaPipe (`mediapipe`)
- **Scientific Computing**: NumPy (`numpy`)
- **Backend Server**: Flask (`flask`)
- **Frontend UI**: HTML5, Custom CSS3 (Glassmorphism & Cyberpunk Theme), JavaScript (ES6+ fetch API), Bootstrap 5, FontAwesome 6, Google Fonts (Orbitron / Outfit)

---

## 🏗️ System Architecture

```
                       ┌──────────────────────┐
                       │    Webcam Input      │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │    CameraManager     │
                       └──────────┬───────────┘
                                  │
            ┌─────────────────────┴─────────────────────┐
            ▼                                           ▼
┌───────────────────────┐                   ┌───────────────────────┐
│     HandTracker       │                   │    EmotionDetector    │
│  (MediaPipe Hands)    │                   │   (OpenCV Face + FE)  │
└───────────┬───────────┘                   └───────────┬───────────┘
            │                                           │
            ▼                                           │
┌───────────────────────┐                               │
│  GestureRecognizer    │                               │
│  (21-Landmark Rules)  │                               │
└───────────┬───────────┘                               │
            │                                           │
            └─────────────────────┬─────────────────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  Flask Web Server    │
                       │   (/video_feed &     │
                       │     /api/data)       │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Cyberpunk Dashboard  │
                       │  (Browser Client)    │
                       └──────────────────────┘
```

---

## 📂 Project Structure

```
AI_Vision_Project/
│
├── app.py                   # Flask server, CameraManager & streaming API routes
├── hand_tracking.py         # MediaPipe hand detector & landmark overlay generator
├── emotion_detection.py     # Real-time face detection & emotion classifier
├── gesture_recognition.py   # Landmark geometry & gesture rule classifier
├── requirements.txt         # Project dependencies manifest
├── README.md                # Technical documentation
│
├── templates/
│   └── index.html           # Futuristic cyber-themed HTML5 dashboard layout
│
└── static/
    ├── style.css            # Custom glassmorphism, neon glows & responsive styles
    └── script.js            # Asynchronous telemetry polling & UI state controller
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+ installed on system.
- A functional webcam or connected camera device.

### 1. Clone or Download Project
Navigated to project folder:
```bash
cd AI_Vision_Project
```

### 2. Install Dependencies
Install required packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Flask server:
```bash
python app.py
```

### 4. Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
Click **"Start Camera"** to launch real-time hand tracking and emotion detection!

---

## ✋ Recognized Gestures

| Gesture | Emoji | Description |
| :--- | :---: | :--- |
| **Open Palm** | ✋ | All 5 fingers extended outwards |
| **Fist** | ✊ | All 5 fingers folded tightly into palm |
| **Thumbs Up** | 👍 | Thumb pointing upwards, remaining 4 fingers folded |
| **Thumbs Down** | 👎 | Thumb pointing downwards, remaining 4 fingers folded |
| **Victory** | ✌️ | Index and Middle fingers extended upwards |
| **OK** | 👌 | Thumb tip and Index tip touching, other 3 fingers extended |
| **One Finger** | ☝️ | Index finger pointing upwards |
| **Three Fingers** | 🤟 | Index, Middle, and Ring/Thumb extended |

---

## 😊 Recognized Emotions

- 😊 **Happy**: High contrast/edge density in mouth & smile region.
- 😢 **Sad**: Low facial contrast & mouth region downward curvature.
- 😡 **Angry**: High eyebrow furrowing intensity.
- 😨 **Fear**: Wide eye region & open mouth expression.
- 😲 **Surprise**: Wide eye region + high mouth opening ratio.
- 😐 **Neutral**: Balanced, relaxed facial expression.
- 🤢 **Disgust**: Asymmetric facial feature tension.

---

## 📷 Screenshots

> *Live webcam stream with cyber landmark overlay, glowing gesture tags, FPS meter, and emotion confidence spectrum.*

---

## 🔮 Future Improvements

- [ ] **Voice Command Controls**: Integrate speech recognition for hands-free dashboard interaction.
- [ ] **Sign-Language Translation**: Expand gesture classes to full ASL alphabet & word phrases.
- [ ] **Emotion-Based Playlist**: Automatically trigger music playlists matching current emotion.
- [ ] **Gesture-Controlled 3D Canvas**: Draw in virtual 3D space using index finger tracking.
- [ ] **Multi-Person Analytics**: Track and aggregate emotion trends across multiple faces simultaneously.

---

## 👤 Author

Developed with ❤️ using **Python, OpenCV, MediaPipe, and Flask**.
