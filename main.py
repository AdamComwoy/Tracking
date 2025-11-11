import time
import cv2
import mediapipe as mp
from typing import Tuple


class ControlPanel:
    """Manages UI controls and trackbars for the application."""

    WINDOW_NAME = "Controls"

    def __init__(self):
        self.window_name = self.WINDOW_NAME
        self._create_window()
        self._create_trackbars()

    def _create_window(self):
        """Creates the control window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 520, 280)

    def _create_trackbars(self):
        """Creates all trackbars with default values."""
        cv2.createTrackbar("Det%", self.window_name, 50, 100, lambda _: None)
        cv2.createTrackbar("Track%", self.window_name, 50, 100, lambda _: None)
        cv2.createTrackbar("Hands", self.window_name, 2, 2, lambda _: None)
        cv2.createTrackbar("Face", self.window_name, 1, 1, lambda _: None)
        cv2.createTrackbar("Scale%", self.window_name, 60, 100, lambda _: None)
        cv2.createTrackbar("Thick", self.window_name, 2, 4, lambda _: None)

    def get_values(self) -> Tuple[float, float, int, int, int, int]:
        """
        Get current trackbar values.

        Returns:
            Tuple of (detection_confidence, tracking_confidence, max_hands,
                     face_enabled, processing_scale, draw_thickness)
        """
        det_c = cv2.getTrackbarPos("Det%", self.window_name) / 100.0
        trk_c = cv2.getTrackbarPos("Track%", self.window_name) / 100.0
        max_hands = max(1, cv2.getTrackbarPos("Hands", self.window_name))
        face_on = cv2.getTrackbarPos("Face", self.window_name)
        proc_scale = max(30, cv2.getTrackbarPos("Scale%", self.window_name))
        draw_th = max(1, cv2.getTrackbarPos("Thick", self.window_name))
        return det_c, trk_c, max_hands, face_on, proc_scale, draw_th


class MediaPipeDetector:
    """Handles MediaPipe hand and face detection."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = None
        self.face = None
        self.current_params = None

    def initialize(self, det_confidence: float, track_confidence: float,
                   max_hands: int, face_enabled: bool):
        """
        Initialize or reinitialize MediaPipe models with given parameters.

        Args:
            det_confidence: Detection confidence threshold (0-1)
            track_confidence: Tracking confidence threshold (0-1)
            max_hands: Maximum number of hands to detect
            face_enabled: Whether to enable face detection
        """
        params = (det_confidence, track_confidence, max_hands, face_enabled)

        # Only reinitialize if parameters changed
        if params == self.current_params:
            return

        # Close existing models
        self.close()

        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=max_hands,
            min_detection_confidence=det_confidence,
            min_tracking_confidence=track_confidence,
        )

        # Initialize face detection if enabled
        self.face = (
            self.mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=det_confidence,
                min_tracking_confidence=track_confidence,
            )
            if face_enabled
            else None
        )

        self.current_params = params

    def process(self, rgb_image):
        """
        Process an RGB image for hand and face detection.

        Args:
            rgb_image: RGB image to process

        Returns:
            Tuple of (hand_results, face_results)
        """
        rgb_image.flags.writeable = False
        hand_results = self.hands.process(rgb_image) if self.hands else None
        face_results = self.face.process(rgb_image) if self.face else None
        rgb_image.flags.writeable = True

        return hand_results, face_results

    def close(self):
        """Close and cleanup MediaPipe models."""
        if self.hands:
            self.hands.close()
            self.hands = None
        if self.face:
            self.face.close()
            self.face = None


class HandGestureAnalyzer:
    """Analyzes hand gestures and counts fingers."""

    @staticmethod
    def count_fingers(landmarks, hand_label: str) -> int:
        """
        Count extended fingers using landmark positions.

        Args:
            landmarks: Hand landmarks from MediaPipe
            hand_label: 'Left' or 'Right'

        Returns:
            Number of extended fingers (0-5)
        """
        if not landmarks or len(landmarks) != 21:
            return 0

        count = 0

        # Count four fingers (index, middle, ring, pinky)
        for tip_id in [8, 12, 16, 20]:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                count += 1

        # Count thumb based on hand orientation
        if hand_label == "Right" and landmarks[4].x > landmarks[3].x:
            count += 1
        elif hand_label == "Left" and landmarks[4].x < landmarks[3].x:
            count += 1

        return count


class Renderer:
    """Handles all rendering and drawing operations."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def put_text(image, text: str, position: Tuple[int, int],
                 scale: float = 0.8, color: Tuple[int, int, int] = (255, 255, 255),
                 thickness: int = 2):
        """Draw text on image."""
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   scale, color, thickness, cv2.LINE_AA)

    def draw_hands(self, frame, hand_results, draw_thickness: int) -> int:
        """
        Draw hand landmarks and finger counts on frame.

        Args:
            frame: Image to draw on
            hand_results: MediaPipe hand detection results
            draw_thickness: Thickness of drawing lines

        Returns:
            Total count of extended fingers across all hands
        """
        total_fingers = 0

        if not hand_results or not hand_results.multi_hand_landmarks:
            return total_fingers

        H, W = frame.shape[:2]

        for landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                         hand_results.multi_handedness):
            hand_label = handedness.classification[0].label
            finger_count = HandGestureAnalyzer.count_fingers(landmarks.landmark, hand_label)
            total_fingers += finger_count

            # Draw hand landmarks
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=draw_thickness,
                                        circle_radius=max(1, draw_thickness - 1)),
                self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=draw_thickness),
            )

            # Draw finger count label
            xs = [int(p.x * W) for p in landmarks.landmark]
            ys = [int(p.y * H) for p in landmarks.landmark]
            x = max(min(xs), 10)
            y = max(min(ys) - 10, 20)
            self.put_text(frame, f"{hand_label}: {finger_count}", (x, y),
                         scale=0.7, color=(0, 255, 255), thickness=2)

        return total_fingers

    def draw_face(self, frame, face_results, draw_thickness: int):
        """
        Draw face mesh on frame.

        Args:
            frame: Image to draw on
            face_results: MediaPipe face detection results
            draw_thickness: Thickness of drawing lines
        """
        if not face_results or not face_results.multi_face_landmarks:
            return

        for face_landmarks in face_results.multi_face_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face.FACEMESH_TESSELATION,
                self.mp_draw.DrawingSpec(color=(0, 150, 255),
                                        thickness=max(1, draw_thickness - 1),
                                        circle_radius=1),
                self.mp_draw.DrawingSpec(color=(255, 255, 255),
                                        thickness=max(1, draw_thickness - 1)),
            )


class HandFaceTracker:
    """Main application class for hand and face tracking."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.control_panel = ControlPanel()
        self.detector = MediaPipeDetector()
        self.renderer = Renderer()

        self.cap = None
        self.fps = 0.0
        self.prev_time = time.time()

    def _initialize_camera(self) -> bool:
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"Kamera sa nedá otvoriť (index {self.camera_index}).")
            return False

        # Set higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        return True

    def _process_frame(self, frame, proc_scale: float):
        """
        Process frame with optional downscaling.

        Args:
            frame: Input BGR frame
            proc_scale: Processing scale percentage (30-100)

        Returns:
            RGB processed frame
        """
        H, W = frame.shape[:2]
        scale = proc_scale / 100.0

        if scale < 1.0:
            processed = cv2.resize(frame, (int(W * scale), int(H * scale)),
                                  interpolation=cv2.INTER_AREA)
        else:
            processed = frame

        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    def _draw_info(self, frame, total_fingers: int):
        """Draw FPS and total finger count on frame."""
        self.renderer.put_text(frame, f"Total: {total_fingers}", (12, 36),
                              scale=1.0, color=(255, 255, 255), thickness=2)
        self.renderer.put_text(frame, f"FPS: {self.fps:.1f}", (12, 68),
                              scale=0.8, color=(0, 200, 255), thickness=2)

    def run(self):
        """Main application loop."""
        if not self._initialize_camera():
            return

        window_name = "Ruky + (voliteľne) Face  —  stlač 'q' pre ukončenie"

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("⚠️  Nedá sa čítať snímok z kamery.")
                    break

                frame = cv2.flip(frame, 1)  # Mirror for selfie view

                # Get control values
                det_c, trk_c, max_hands, face_on, proc_scale, draw_th = self.control_panel.get_values()

                # Initialize/reinitialize detector if needed
                self.detector.initialize(det_c, trk_c, max_hands, face_on)

                # Process frame
                rgb = self._process_frame(frame, proc_scale)
                hand_results, face_results = self.detector.process(rgb)

                # Render results
                total_fingers = self.renderer.draw_hands(frame, hand_results, draw_th)
                self.renderer.draw_face(frame, face_results, draw_th)

                # Update and draw info
                self._update_fps()
                self._draw_info(frame, total_fingers)

                # Display frame
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        self.detector.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main(camera_index: int = 0):
    """Entry point for the application."""
    tracker = HandFaceTracker(camera_index)
    tracker.run()


if __name__ == "__main__":
    main()
