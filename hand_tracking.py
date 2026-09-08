import os
import cv2
import urllib.request
import numpy as np
import mediapipe as mp
from gesture_recognition import GestureRecognizer

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

class HandTracker:
    """
    Real-time Hand Detector and Landmark Tracker powered by MediaPipe Tasks API.
    Detects up to `max_hands`, extracts 21 landmarks, calculates finger states,
    recognizes gestures, and draws futuristic visual overlays.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"

    def __init__(self, max_hands=2, detection_con=0.5, track_con=0.5):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.gesture_recognizer = GestureRecognizer()

        # Download model task file if not present locally
        self._ensure_model_exists()

        # Initialize MediaPipe Tasks HandLandmarker
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading MediaPipe HandLandmarker model file to '{self.MODEL_PATH}'...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                print("HandLandmarker model download complete!")
            except Exception as e:
                print("Model download error:", e)

    def process(self, frame, draw=True):
        """
        Processes an OpenCV image frame.
        
        Returns:
            annotated_frame (np.ndarray): OpenCV image with visual overlays.
            hand_results (list[dict]): Extracted hand metrics.
            total_fingers (int): Total raised fingers across all hands.
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.landmarker.detect(mp_image)

        hand_data = []
        total_fingers = 0

        if results and results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Extract handedness label (Left/Right)
                handedness_label = "Right"
                if results.handedness and idx < len(results.handedness):
                    raw_label = results.handedness[idx][0].category_name
                    # Flip label for selfie-mirrored perspective
                    handedness_label = "Left" if raw_label == "Right" else "Right"

                # Extract landmark coordinates
                landmarks_list = []
                x_coords = []
                y_coords = []

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_list.append((cx, cy, lm.z))
                    x_coords.append(cx)
                    y_coords.append(cy)

                # Bounding box coordinates
                xmin, xmax = max(0, min(x_coords) - 15), min(w, max(x_coords) + 15)
                ymin, ymax = max(0, min(y_coords) - 15), min(h, max(y_coords) + 15)

                # Calculate finger states and raised count
                finger_states = GestureRecognizer.get_finger_states(landmarks_list, handedness_label)
                raised_count = sum(finger_states)
                total_fingers += raised_count

                # Recognize gesture
                gesture_name, emoji, confidence = self.gesture_recognizer.recognize(
                    landmarks_list, handedness_label
                )

                hand_info = {
                    "hand_index": idx,
                    "handedness": handedness_label,
                    "landmarks": landmarks_list,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "finger_states": finger_states,
                    "raised_count": raised_count,
                    "gesture": gesture_name,
                    "emoji": emoji,
                    "confidence": round(confidence * 100, 1)
                }
                hand_data.append(hand_info)

                if draw:
                    # Draw futuristic Cyber Hand Skeleton
                    self._draw_cyber_overlay(frame, landmarks_list, xmin, ymin, xmax, ymax, handedness_label, gesture_name, emoji, raised_count, w, h)

        return frame, hand_data, total_fingers

    def _draw_cyber_overlay(self, frame, landmarks, xmin, ymin, xmax, ymax, handedness, gesture, emoji, raised_count, w, h):
        """Renders futuristic glowing cyber overlay on detected hand."""

        primary_color = (255, 230, 0) if handedness == "Right" else (255, 0, 230)
        joint_color = (0, 255, 128)

        # 1. Custom landmark line connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (landmarks[start_idx][0], landmarks[start_idx][1])
            p2 = (landmarks[end_idx][0], landmarks[end_idx][1])
            cv2.line(frame, p1, p2, primary_color, 2, cv2.LINE_AA)

        # 2. Draw Joint Nodes
        for p in landmarks:
            cv2.circle(frame, (p[0], p[1]), 4, joint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (p[0], p[1]), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Rounded Bounding Box with Corner Accents
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), primary_color, 1, cv2.LINE_AA)

        corner_len = min(20, (xmax - xmin) // 4)
        if corner_len > 0:
            # Top-Left
            cv2.line(frame, (xmin, ymin), (xmin + corner_len, ymin), primary_color, 3)
            cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_len), primary_color, 3)
            # Top-Right
            cv2.line(frame, (xmax, ymin), (xmax - corner_len, ymin), primary_color, 3)
            cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_len), primary_color, 3)
            # Bottom-Left
            cv2.line(frame, (xmin, ymax), (xmin + corner_len, ymax), primary_color, 3)
            cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_len), primary_color, 3)
            # Bottom-Right
            cv2.line(frame, (xmax, ymax), (xmax - corner_len, ymax), primary_color, 3)
            cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_len), primary_color, 3)

        # 4. Label Badge ABOVE hand box
        label_text = f"{handedness}: {gesture} ({raised_count})"
        badge_y = max(ymin - 10, 25)

        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        cv2.rectangle(
            frame,
            (xmin, badge_y - text_h - 8),
            (xmin + text_w + 16, badge_y + 4),
            (15, 23, 42),
            -1
        )
        cv2.rectangle(
            frame,
            (xmin, badge_y - text_h - 8),
            (xmin + text_w + 16, badge_y + 4),
            primary_color,
            1
        )

        cv2.putText(
            frame,
            label_text,
            (xmin + 8, badge_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
