import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify, request
from hand_tracking import HandTracker
from emotion_detection import EmotionDetector

app = Flask(__name__)

class CameraManager:
    """
    Thread-safe Camera and Computer Vision Pipeline Manager.
    Handles video acquisition, hand tracking, emotion detection, FPS tracking,
    and graceful hardware resource lifecycle management.
    """

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Core CV Engines
        self.hand_tracker = None
        self.emotion_detector = None

        # Telemetry State
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.telemetry = {
            "camera_active": False,
            "fps": 0.0,
            "hands_detected": 0,
            "total_fingers": 0,
            "hands": [],
            "emotion": {
                "detected": False,
                "primary": "Neutral",
                "emoji": "😐",
                "confidence": 0.0,
                "all_emotions": {e: 0.0 for e in EmotionDetector.EMOTIONS}
            }
        }

    def start_camera(self, camera_index=0):
        """Starts webcam capture stream and initializes AI models."""
        with self.lock:
            if self.is_running:
                return True, "Camera is already running."

            # Attempt to open capture device
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                # Fallback to default index without specific backend driver
                self.cap = cv2.VideoCapture(camera_index)

            if not self.cap.isOpened():
                return False, "Unable to access webcam. Please check camera permissions and hardware connection."

            # Set resolution & FPS optimizations
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Lazy initialize CV processors
            if self.hand_tracker is None:
                self.hand_tracker = HandTracker(max_hands=2, detection_con=0.6, track_con=0.6)
            if self.emotion_detector is None:
                self.emotion_detector = EmotionDetector(process_every_n_frames=3)

            self.is_running = True
            self.last_frame_time = time.time()
            self.telemetry["camera_active"] = True
            return True, "Camera started successfully."

    def stop_camera(self):
        """Stops capture stream and releases video device."""
        with self.lock:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.telemetry["camera_active"] = False
            self.telemetry["fps"] = 0.0
            return True, "Camera stopped successfully."

    def get_processed_frame(self):
        """
        Captures single frame, executes Hand Tracking + Emotion Detection combined,
        updates telemetry, and returns JPEG encoded image bytes.
        """
        if not self.is_running or self.cap is None:
            return None

        with self.lock:
            if not self.cap.isOpened():
                return None
            success, frame = self.cap.read()

        if not success or frame is None:
            return None

        # Flip horizontally for intuitive selfie view
        frame = cv2.flip(frame, 1)

        # 1. Process Hand Tracking
        frame, hand_data, total_fingers = self.hand_tracker.process(frame, draw=True)

        # 2. Process Emotion Detection
        frame, emotion_data = self.emotion_detector.detect(frame, draw=True)

        # 3. Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = round(0.8 * self.fps + 0.2 * current_fps, 1)  # Smooth FPS estimate
        self.last_frame_time = current_time

        # 4. Draw HUD Dashboard Header on Video Stream
        self._draw_video_dashboard_hud(frame, hand_data, total_fingers, emotion_data, self.fps)

        # 5. Update Telemetry state
        self.telemetry = {
            "camera_active": True,
            "fps": self.fps,
            "hands_detected": len(hand_data),
            "total_fingers": total_fingers,
            "hands": [
                {
                    "handedness": h["handedness"],
                    "gesture": h["gesture"],
                    "emoji": h["emoji"],
                    "raised_count": h["raised_count"],
                    "confidence": h["confidence"]
                }
                for h in hand_data
            ],
            "emotion": {
                "detected": emotion_data["detected"],
                "primary": emotion_data["emotion"],
                "emoji": emotion_data["emoji"],
                "confidence": emotion_data["confidence"],
                "all_emotions": emotion_data["all_emotions"]
            }
        }

        # Encode frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            return None

        return jpeg.tobytes()

    def _draw_video_dashboard_hud(self, frame, hand_data, total_fingers, emotion_data, fps):
        """Draws top Cyber Dashboard Overlay on video frame."""
        h, w, c = frame.shape

        # Semi-transparent Top HUD Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (10, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 48), (w, 48), (0, 240, 255), 1)

        # FPS Counter
        fps_text = f"FPS: {fps}"
        cv2.putText(frame, fps_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)

        # Emotion Status
        emo_text = f"Emotion: {emotion_data['emoji']} {emotion_data['emotion']}"
        cv2.putText(frame, emo_text, (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Hands & Finger Count
        hands_text = f"Hands: {len(hand_data)} | Raised Fingers: {total_fingers}"
        cv2.putText(frame, hands_text, (w - 380, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2, cv2.LINE_AA)


# Global Camera Manager instance
camera_manager = CameraManager()


@app.route('/')
def index():
    """Renders main AI Vision Dashboard HTML."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG streaming route."""
    def generate():
        while True:
            if not camera_manager.is_running:
                # Return placeholder empty state
                time.sleep(0.1)
                continue
            
            frame_bytes = camera_manager.get_processed_frame()
            if frame_bytes is None:
                time.sleep(0.03)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def api_data():
    """Endpoint for returning live telemetry metrics."""
    return jsonify(camera_manager.telemetry)


@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    """Endpoint to trigger camera activation."""
    success, msg = camera_manager.start_camera()
    return jsonify({"success": success, "message": msg})


@app.route('/api/stop_camera', methods=['POST'])
def api_stop_camera():
    """Endpoint to deactivate camera."""
    success, msg = camera_manager.stop_camera()
    return jsonify({"success": success, "message": msg})


if __name__ == '__main__':
    # Default Flask local dev server
    print("\n========================================================")
    print(" [AI] AI Vision: Hand Tracking & Emotion Detection Server")
    print(" Open URL in browser: http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
