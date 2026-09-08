import math
import numpy as np

class GestureRecognizer:
    """
    Recognizes hand gestures from 21 MediaPipe hand landmarks and finger states.
    Supported gestures:
    - ✋ Open Palm
    - ✊ Fist
    - 👍 Thumbs Up
    - 👎 Thumbs Down
    - ✌️ Victory
    - 👌 OK
    - ☝️ One Finger
    - 🤟 Three Fingers
    - ❓ Unknown
    """

    def __init__(self):
        pass

    @staticmethod
    def _euclidean_distance(p1, p2):
        """Calculate 2D Euclidean distance between two landmark points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def get_finger_states(landmarks, handedness="Right"):
        """
        Determines which fingers are open (extended).
        Returns a list of booleans: [thumb, index, middle, ring, pinky]
        True = extended, False = folded.
        """
        if not landmarks or len(landmarks) < 21:
            return [False, False, False, False, False]

        wrist = landmarks[0]

        # Index: 8 (tip), 6 (pip)
        # Middle: 12 (tip), 10 (pip)
        # Ring: 16 (tip), 14 (pip)
        # Pinky: 20 (tip), 18 (pip)
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        states = []

        # Thumb detection
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        pinky_mcp = landmarks[17]

        dist_thumb_pinky = GestureRecognizer._euclidean_distance(thumb_tip, pinky_mcp)
        dist_mcp_pinky = GestureRecognizer._euclidean_distance(thumb_mcp, pinky_mcp)

        if handedness == "Right":
            thumb_open = thumb_tip[0] < thumb_ip[0] or dist_thumb_pinky > dist_mcp_pinky * 1.2
        else:
            thumb_open = thumb_tip[0] > thumb_ip[0] or dist_thumb_pinky > dist_mcp_pinky * 1.2

        states.append(thumb_open)

        # 4 fingers: Check if tip is further from wrist than PIP joint
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_dist = GestureRecognizer._euclidean_distance(landmarks[tip_idx], wrist)
            pip_dist = GestureRecognizer._euclidean_distance(landmarks[pip_idx], wrist)
            states.append(tip_dist > pip_dist)

        return states

    def recognize(self, landmarks, handedness="Right"):
        """
        Recognize gesture given 21 landmarks [(x, y, z), ...].
        Returns tuple: (gesture_name, emoji, confidence)
        """
        if not landmarks or len(landmarks) < 21:
            return "Unknown", "❓", 0.0

        finger_states = self.get_finger_states(landmarks, handedness)
        thumb, index, middle, ring, pinky = finger_states

        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]

        # Scale reference distance (Wrist to Middle MCP)
        hand_scale = self._euclidean_distance(wrist, landmarks[9])
        if hand_scale < 1e-5:
            hand_scale = 1.0

        # Check OK gesture
        dist_thumb_index = self._euclidean_distance(thumb_tip, index_tip) / hand_scale
        if dist_thumb_index < 0.25 and middle and ring and pinky:
            return "OK", "👌", 0.95

        # Check Thumbs Up
        if thumb and not index and not middle and not ring and not pinky:
            if thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]:
                return "Thumbs Up", "👍", 0.96

        # Check Thumbs Down
        if thumb and not index and not middle and not ring and not pinky:
            if thumb_tip[1] > thumb_ip[1] > thumb_mcp[1]:
                return "Thumbs Down", "👎", 0.96

        # Count total extended fingers
        raised_count = sum(finger_states)

        # Open Palm
        if raised_count == 5 or (not thumb and index and middle and ring and pinky):
            return "Open Palm", "✋", 0.98

        # Fist
        if raised_count == 0 or (thumb and not index and not middle and not ring and not pinky and dist_thumb_index < 0.3):
            return "Fist", "✊", 0.98

        # Victory
        if index and middle and not ring and not pinky:
            return "Victory", "✌️", 0.96

        # One Finger
        if index and not middle and not ring and not pinky:
            return "One Finger", "☝️", 0.95

        # Three Fingers
        if (index and middle and ring and not pinky) or (thumb and index and middle and not ring and not pinky):
            return "Three Fingers", "🤟", 0.94

        # Fallbacks
        if raised_count == 1:
            if index:
                return "One Finger", "☝️", 0.85
            return "1 Finger Raised", "☝️", 0.80
        elif raised_count == 2:
            return "Victory", "✌️", 0.80
        elif raised_count == 3:
            return "Three Fingers", "🤟", 0.85
        elif raised_count == 4:
            return "4 Fingers Raised", "✋", 0.85

        return "Active Hand", "🖐️", 0.70
