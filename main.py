import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings."""
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    max_hands: int = 2
    face_enabled: bool = True
    processing_scale: int = 60
    draw_thickness: int = 2
    deepfake_enabled: bool = False
    deepfake_mode: int = 0  # 0=blur, 1=pixelate, 2=swap


class UnifiedUI:
    """Manages unified UI with video and side panel controls."""

    def __init__(self, video_width: int = 1280, video_height: int = 720):
        self.video_width = video_width
        self.video_height = video_height
        self.control_width = 350
        self.total_width = video_width + self.control_width
        self.total_height = video_height
        
        self.window_name = "Hand & Face Tracker with Deepfake"
        self.config = AppConfig()
        
        self._create_window()
        self._create_trackbars()

    def _create_window(self):
        """Creates the unified window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.total_width, self.total_height)

    def _create_trackbars(self):
        """Creates all trackbars with default values."""
        cv2.createTrackbar("Detection %", self.window_name, 50, 100, self._update_config)
        cv2.createTrackbar("Tracking %", self.window_name, 50, 100, self._update_config)
        cv2.createTrackbar("Max Hands", self.window_name, 2, 4, self._update_config)
        cv2.createTrackbar("Face Mesh", self.window_name, 1, 1, self._update_config)
        cv2.createTrackbar("Scale %", self.window_name, 60, 100, self._update_config)
        cv2.createTrackbar("Thickness", self.window_name, 2, 5, self._update_config)
        cv2.createTrackbar("Deepfake ON", self.window_name, 0, 1, self._update_config)
        cv2.createTrackbar("Deepfake Mode", self.window_name, 0, 2, self._update_config)

    def _update_config(self, _):
        """Update configuration from trackbar values."""
        self.config.detection_confidence = cv2.getTrackbarPos("Detection %", self.window_name) / 100.0
        self.config.tracking_confidence = cv2.getTrackbarPos("Tracking %", self.window_name) / 100.0
        self.config.max_hands = max(1, cv2.getTrackbarPos("Max Hands", self.window_name))
        self.config.face_enabled = bool(cv2.getTrackbarPos("Face Mesh", self.window_name))
        self.config.processing_scale = max(30, cv2.getTrackbarPos("Scale %", self.window_name))
        self.config.draw_thickness = max(1, cv2.getTrackbarPos("Thickness", self.window_name))
        self.config.deepfake_enabled = bool(cv2.getTrackbarPos("Deepfake ON", self.window_name))
        self.config.deepfake_mode = cv2.getTrackbarPos("Deepfake Mode", self.window_name)

    def create_control_panel(self, info_text: dict) -> np.ndarray:
        """
        Create the side control panel with current settings.
        
        Args:
            info_text: Dictionary with current application info
            
        Returns:
            Control panel image
        """
        panel = np.zeros((self.total_height, self.control_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        y_offset = 30
        line_height = 35
        
        # Title
        cv2.putText(panel, "CONTROLS", (20, y_offset), 
                   cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # Draw separator
        cv2.line(panel, (20, y_offset), (self.control_width - 20, y_offset), 
                (100, 100, 100), 1)
        y_offset += 20

        # Settings display
        settings = [
            ("Detection", f"{self.config.detection_confidence:.0%}"),
            ("Tracking", f"{self.config.tracking_confidence:.0%}"),
            ("Max Hands", f"{self.config.max_hands}"),
            ("Face Mesh", "ON" if self.config.face_enabled else "OFF"),
            ("Scale", f"{self.config.processing_scale}%"),
            ("Thickness", f"{self.config.draw_thickness}"),
            ("", ""),  # Spacer
            ("Deepfake", "ON" if self.config.deepfake_enabled else "OFF"),
            ("Mode", ["Blur", "Pixelate", "Swap"][self.config.deepfake_mode]),
        ]

        for label, value in settings:
            if label:  # Skip spacers
                cv2.putText(panel, label, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(panel, str(value), (200, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_offset += line_height

        # Draw separator
        y_offset += 10
        cv2.line(panel, (20, y_offset), (self.control_width - 20, y_offset), 
                (100, 100, 100), 1)
        y_offset += 30

        # Info display
        cv2.putText(panel, "INFO", (20, y_offset), 
                   cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)
        y_offset += line_height

        info_items = [
            ("FPS", f"{info_text.get('fps', 0):.1f}"),
            ("Fingers", str(info_text.get('fingers', 0))),
            ("Hands", str(info_text.get('hands', 0))),
            ("Faces", str(info_text.get('faces', 0))),
        ]

        for label, value in info_items:
            cv2.putText(panel, label, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, str(value), (200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            y_offset += line_height

        # Instructions at bottom
        y_offset = self.total_height - 100
        cv2.putText(panel, "CONTROLS:", (20, y_offset), 
                   cv2.FONT_HERSHEY_BOLD, 0.6, (255, 200, 100), 1)
        y_offset += 25
        
        instructions = [
            "Q - Quit",
            "S - Screenshot",
            "R - Reset"
        ]
        
        for instruction in instructions:
            cv2.putText(panel, instruction, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_offset += 22

        return panel

    def combine_frame_and_panel(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """
        Combine video frame with control panel.
        
        Args:
            frame: Video frame
            info: Info dictionary for panel
            
        Returns:
            Combined image
        """
        # Resize frame if needed
        if frame.shape[:2] != (self.video_height, self.video_width):
            frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        # Create panel
        panel = self.create_control_panel(info)
        
        # Combine horizontally
        combined = np.hstack([frame, panel])
        
        return combined


class MediaPipeDetector:
    """Handles MediaPipe hand and face detection."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = None
        self.face = None
        self.current_params = None

    def initialize(self, config: AppConfig):
        """
        Initialize or reinitialize MediaPipe models with given parameters.
        
        Args:
            config: Application configuration
        """
        params = (config.detection_confidence, config.tracking_confidence, 
                 config.max_hands, config.face_enabled)

        # Only reinitialize if parameters changed
        if params == self.current_params:
            return

        # Close existing models
        self.close()

        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=config.max_hands,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )

        # Initialize face detection if enabled
        self.face = (
            self.mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=config.detection_confidence,
                min_tracking_confidence=config.tracking_confidence,
            )
            if config.face_enabled
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


class DeepfakeProcessor:
    """Handles deepfake/face manipulation effects."""

    @staticmethod
    def get_face_bbox(face_landmarks, image_shape) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding box from face landmarks.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image_shape: Shape of the image (H, W, C)
            
        Returns:
            Tuple of (x, y, w, h) or None
        """
        if not face_landmarks:
            return None

        H, W = image_shape[:2]
        
        # Get all landmark coordinates
        xs = [int(lm.x * W) for lm in face_landmarks.landmark]
        ys = [int(lm.y * H) for lm in face_landmarks.landmark]
        
        # Calculate bounding box with some padding
        x_min, x_max = max(0, min(xs) - 20), min(W, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 30), min(H, max(ys) + 20)
        
        return x_min, y_min, x_max - x_min, y_max - y_min

    @staticmethod
    def apply_blur(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                   intensity: int = 35) -> np.ndarray:
        """
        Apply blur effect to face region.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            intensity: Blur kernel size (must be odd)
            
        Returns:
            Frame with blurred face
        """
        x, y, w, h = bbox
        
        # Make intensity odd
        intensity = intensity if intensity % 2 == 1 else intensity + 1
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (intensity, intensity), 0)
        
        # Replace face region
        frame[y:y+h, x:x+w] = blurred_face
        
        return frame

    @staticmethod
    def apply_pixelate(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                       pixel_size: int = 15) -> np.ndarray:
        """
        Apply pixelation effect to face region.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            pixel_size: Size of pixels
            
        Returns:
            Frame with pixelated face
        """
        x, y, w, h = bbox
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        # Calculate downscale size
        small_h, small_w = max(1, h // pixel_size), max(1, w // pixel_size)
        
        # Downscale then upscale
        small = cv2.resize(face_region, (small_w, small_h), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), 
                              interpolation=cv2.INTER_NEAREST)
        
        # Replace face region
        frame[y:y+h, x:x+w] = pixelated
        
        return frame

    @staticmethod
    def apply_face_swap(frame: np.ndarray, face_landmarks, 
                       stored_face: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simple face swap effect (stores first face, swaps with subsequent ones).
        
        Args:
            frame: Input frame
            face_landmarks: MediaPipe face landmarks
            stored_face: Previously stored face region
            
        Returns:
            Tuple of (modified frame, stored face)
        """
        bbox = DeepfakeProcessor.get_face_bbox(face_landmarks, frame.shape)
        if not bbox:
            return frame, stored_face
            
        x, y, w, h = bbox
        current_face = frame[y:y+h, x:x+w].copy()
        
        # If no stored face, store current and return
        if stored_face is None:
            return frame, current_face
        
        # Resize stored face to current face size and swap
        try:
            swapped = cv2.resize(stored_face, (w, h))
            frame[y:y+h, x:x+w] = swapped
        except:
            pass  # Skip if resize fails
            
        return frame, stored_face

    def process_face(self, frame: np.ndarray, face_results, mode: int,
                    stored_face: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply deepfake effect based on mode.
        
        Args:
            frame: Input frame
            face_results: MediaPipe face detection results
            mode: Effect mode (0=blur, 1=pixelate, 2=swap)
            stored_face: Stored face for swap mode
            
        Returns:
            Tuple of (processed frame, updated stored face)
        """
        if not face_results or not face_results.multi_face_landmarks:
            return frame, stored_face

        for face_landmarks in face_results.multi_face_landmarks:
            bbox = self.get_face_bbox(face_landmarks, frame.shape)
            
            if bbox:
                if mode == 0:  # Blur
                    frame = self.apply_blur(frame, bbox, intensity=45)
                elif mode == 1:  # Pixelate
                    frame = self.apply_pixelate(frame, bbox, pixel_size=20)
                elif mode == 2:  # Face swap
                    frame, stored_face = self.apply_face_swap(frame, face_landmarks, stored_face)

        return frame, stored_face


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

    def draw_hands(self, frame, hand_results, draw_thickness: int) -> Tuple[int, int]:
        """
        Draw hand landmarks and finger counts on frame.
        
        Args:
            frame: Image to draw on
            hand_results: MediaPipe hand detection results
            draw_thickness: Thickness of drawing lines
            
        Returns:
            Tuple of (total_fingers, num_hands)
        """
        total_fingers = 0
        num_hands = 0

        if not hand_results or not hand_results.multi_hand_landmarks:
            return total_fingers, num_hands

        H, W = frame.shape[:2]
        num_hands = len(hand_results.multi_hand_landmarks)

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

        return total_fingers, num_hands

    def draw_face(self, frame, face_results, draw_thickness: int) -> int:
        """
        Draw face mesh on frame.
        
        Args:
            frame: Image to draw on
            face_results: MediaPipe face detection results
            draw_thickness: Thickness of drawing lines
            
        Returns:
            Number of faces detected
        """
        num_faces = 0
        
        if not face_results or not face_results.multi_face_landmarks:
            return num_faces

        num_faces = len(face_results.multi_face_landmarks)

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
        
        return num_faces


class HandFaceTracker:
    """Main application class for hand and face tracking with deepfake effects."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.ui = UnifiedUI()
        self.detector = MediaPipeDetector()
        self.deepfake = DeepfakeProcessor()
        self.renderer = Renderer()

        self.cap = None
        self.fps = 0.0
        self.prev_time = time.time()
        self.stored_face = None
        self.screenshot_counter = 0

    def _initialize_camera(self) -> bool:
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Cannot open camera (index {self.camera_index}).")
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

    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot."""
        filename = f"screenshot_{self.screenshot_counter:04d}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        self.screenshot_counter += 1

    def run(self):
        """Main application loop."""
        if not self._initialize_camera():
            return

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("⚠️  Cannot read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)  # Mirror for selfie view

                # Get current config
                config = self.ui.config

                # Initialize/reinitialize detector if needed
                self.detector.initialize(config)

                # Process frame
                rgb = self._process_frame(frame, config.processing_scale)
                hand_results, face_results = self.detector.process(rgb)

                # Apply deepfake effects if enabled
                if config.deepfake_enabled and face_results:
                    frame, self.stored_face = self.deepfake.process_face(
                        frame, face_results, config.deepfake_mode, self.stored_face
                    )

                # Render results
                total_fingers, num_hands = self.renderer.draw_hands(
                    frame, hand_results, config.draw_thickness
                )
                num_faces = self.renderer.draw_face(
                    frame, face_results, config.draw_thickness
                )

                # Update FPS
                self._update_fps()

                # Create info dictionary
                info = {
                    'fps': self.fps,
                    'fingers': total_fingers,
                    'hands': num_hands,
                    'faces': num_faces
                }

                # Combine frame with control panel
                display = self.ui.combine_frame_and_panel(frame, info)

                # Display
                cv2.imshow(self.ui.window_name, display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(display)
                elif key == ord('r'):
                    self.stored_face = None
                    print("Reset: Stored face cleared")

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
    print("=" * 60)
    print("Hand & Face Tracker with Deepfake Effects")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit application")
    print("  S - Save screenshot")
    print("  R - Reset stored face (for face swap mode)")
    print("\nDeepfake Modes:")
    print("  0 - Blur face")
    print("  1 - Pixelate face")
    print("  2 - Face swap (stores first face, swaps with others)")
    print("=" * 60)
    
    tracker = HandFaceTracker(camera_index)
    tracker.run()


if __name__ == "__main__":
    main()