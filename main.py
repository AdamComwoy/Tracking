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
    face_skeleton_visible: bool = True
    processing_scale: int = 60
    draw_thickness: int = 2
    deepfake_enabled: bool = False
    deepfake_mode: int = 0  # 0=blur, 1=pixelate, 2=swap


class UnifiedUI:
    """Manages UI with separate video and control panel windows."""

    def __init__(self, video_width: int = 1920, video_height: int = 1080):
        self.video_width = video_width
        self.video_height = video_height
        self.control_width = 400  # Initial size
        self.control_height = 950  # Increased for sliders
        self.min_control_width = 300
        self.min_control_height = 700

        self.video_window = "Hand & Face Tracker - Video Feed"
        self.control_window = "Control Dashboard"
        self.config = AppConfig()

        # Slider state management
        self.sliders = {}
        self.active_slider = None
        self.mouse_down = False

        self._create_windows()
        self._setup_mouse_callback()

    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw filled rectangle
        if thickness == -1:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # Draw circles at corners
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Draw rectangle outline
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            # Draw arcs at corners
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    @staticmethod
    def draw_status_indicator(img, center, radius, is_on, scale=1.0):
        """Draw a colored status indicator dot."""
        radius = int(radius * scale)
        if is_on:
            # Green glowing effect
            cv2.circle(img, center, radius + 2, (0, 100, 0), -1)
            cv2.circle(img, center, radius, (50, 255, 50), -1)
        else:
            # Gray inactive
            cv2.circle(img, center, radius + 2, (40, 40, 40), -1)
            cv2.circle(img, center, radius, (100, 100, 100), -1)

    def _create_windows(self):
        """Creates separate windows for video and controls."""
        # Video window - full size, resizable
        cv2.namedWindow(self.video_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.video_window, self.video_width, self.video_height)

        # Control window - resizable
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.control_window, self.control_width, self.control_height)

    def _setup_mouse_callback(self):
        """Setup mouse callback for slider interaction."""
        cv2.setMouseCallback(self.control_window, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for slider interaction."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            # Check if click is on any slider
            for name, slider_info in self.sliders.items():
                sx, sy, sw, sh = slider_info['rect']
                if sx <= x <= sx + sw and sy - 10 <= y <= sy + sh + 10:
                    self.active_slider = name
                    self._update_slider_value(name, x, slider_info)
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_down and self.active_slider:
                slider_info = self.sliders[self.active_slider]
                self._update_slider_value(self.active_slider, x, slider_info)

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            self.active_slider = None

    def _update_slider_value(self, name, mouse_x, slider_info):
        """Update slider value based on mouse position."""
        sx, sy, sw, sh = slider_info['rect']
        min_val, max_val = slider_info['range']

        # Calculate normalized value (0.0 to 1.0)
        normalized = max(0.0, min(1.0, (mouse_x - sx) / sw))

        # Calculate actual value
        if slider_info.get('integer', False):
            value = int(min_val + normalized * (max_val - min_val))
        else:
            value = min_val + normalized * (max_val - min_val)

        # Update config based on slider name
        if name == "Detection":
            self.config.detection_confidence = value
        elif name == "Tracking":
            self.config.tracking_confidence = value
        elif name == "Max Hands":
            self.config.max_hands = max(1, value)
        elif name == "Scale":
            self.config.processing_scale = max(30, value)
        elif name == "Thickness":
            self.config.draw_thickness = max(1, value)
        elif name == "Face Mesh":
            self.config.face_enabled = value > 0.5
        elif name == "Face Skeleton":
            self.config.face_skeleton_visible = value > 0.5
        elif name == "Deepfake":
            self.config.deepfake_enabled = value > 0.5
        elif name == "DF Mode":
            self.config.deepfake_mode = value

    def draw_slider(self, panel, x, y, width, height, value, min_val, max_val,
                   label, base_scale, name, integer=False):
        """
        Draw an interactive slider.

        Args:
            panel: Image to draw on
            x, y: Top-left position
            width, height: Slider dimensions
            value: Current value
            min_val, max_val: Value range
            label: Slider label
            base_scale: Scale factor for responsive design
            name: Slider identifier
            integer: Whether value is integer
        """
        # Store slider info for mouse interaction
        self.sliders[name] = {
            'rect': (x, y, width, height),
            'range': (min_val, max_val),
            'integer': integer
        }

        font_scale = 0.45 * base_scale
        thickness = max(1, int(1 * base_scale))

        # Draw label
        cv2.putText(panel, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

        # Draw slider track (background)
        track_color = (60, 60, 65)
        cv2.rectangle(panel, (x, y), (x + width, y + height), track_color, -1)

        # Calculate fill width based on value
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        fill_width = int(width * normalized)

        # Draw filled portion (active)
        if fill_width > 0:
            fill_color = (100, 180, 255)
            cv2.rectangle(panel, (x, y), (x + fill_width, y + height), fill_color, -1)

        # Draw slider handle
        handle_x = x + fill_width
        handle_color = (150, 220, 255) if self.active_slider == name else (200, 200, 200)
        cv2.circle(panel, (handle_x, y + height // 2), int(8 * base_scale), handle_color, -1)
        cv2.circle(panel, (handle_x, y + height // 2), int(9 * base_scale), (255, 255, 255), 1)

        # Draw current value
        if integer:
            value_text = str(int(value))
        else:
            value_text = f"{value:.0%}" if max_val <= 1.0 else f"{value:.0f}%"

        cv2.putText(panel, value_text, (x + width + 10, y + height - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (120, 200, 255),
                   max(1, int(2 * base_scale)))

    def draw_toggle(self, panel, x, y, width, height, value, label, base_scale, name):
        """Draw an interactive toggle button."""
        # Store as slider for interaction
        self.sliders[name] = {
            'rect': (x, y, width, height),
            'range': (0, 1),
            'integer': False
        }

        font_scale = 0.45 * base_scale
        thickness = max(1, int(1 * base_scale))

        # Draw label
        cv2.putText(panel, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

        # Toggle switch background
        toggle_width = int(50 * base_scale)
        toggle_height = int(24 * base_scale)
        toggle_x = x
        toggle_y = y

        bg_color = (50, 200, 100) if value else (70, 70, 70)
        self.draw_rounded_rect(panel, (toggle_x, toggle_y),
                              (toggle_x + toggle_width, toggle_y + toggle_height),
                              bg_color, -1, int(12 * base_scale))

        # Toggle handle
        handle_radius = int(10 * base_scale)
        handle_x = toggle_x + toggle_width - handle_radius - 4 if value else toggle_x + handle_radius + 4
        handle_y = toggle_y + toggle_height // 2
        cv2.circle(panel, (handle_x, handle_y), handle_radius, (255, 255, 255), -1)

        # Status text
        status_text = "ON" if value else "OFF"
        status_color = (100, 255, 100) if value else (150, 150, 150)
        cv2.putText(panel, status_text, (toggle_x + toggle_width + 10, toggle_y + toggle_height - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color,
                   max(1, int(2 * base_scale)))

    def create_control_panel(self, info_text: dict, width: int = None, height: int = None) -> np.ndarray:
        """
        Create the control panel dashboard with interactive sliders and modern UI.

        Args:
            info_text: Dictionary with current application info
            width: Panel width (None = use default)
            height: Panel height (None = use default)

        Returns:
            Control panel image
        """
        # Clear sliders for redraw
        self.sliders.clear()

        # Use provided dimensions or defaults
        panel_width = max(width or self.control_width, self.min_control_width)
        panel_height = max(height or self.control_height, self.min_control_height)

        # Scale factors for responsive design
        scale_x = panel_width / 400.0
        scale_y = panel_height / 950.0
        base_scale = min(scale_x, scale_y)

        # Responsive sizing
        font_scale_title = 0.7 * base_scale
        font_scale_section = 0.6 * base_scale
        font_scale_text = 0.5 * base_scale
        font_scale_small = 0.45 * base_scale

        thickness_bold = max(1, int(2 * base_scale))
        thickness_normal = max(1, int(1 * base_scale))

        slider_height = int(12 * base_scale)
        line_spacing = int(35 * scale_y)
        margin_x = int(20 * scale_x)
        card_padding = int(15 * scale_x)
        slider_width = int((panel_width - 2 * margin_x - 2 * card_padding - 60) * 0.85)

        # Modern dark background with gradient effect
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        for i in range(panel_height):
            blend = i / panel_height
            color = int(25 + blend * 15)
            panel[i, :] = (color, color, color)

        y_offset = int(25 * scale_y)

        # Title with glow effect
        title_y = y_offset
        cv2.putText(panel, "CONTROL DASHBOARD", (margin_x + 2, title_y + 2),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_title, (30, 30, 30), thickness_bold + 1)
        cv2.putText(panel, "CONTROL DASHBOARD", (margin_x, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_title, (255, 255, 255), thickness_bold)
        y_offset += int(45 * scale_y)

        # Settings Card with Interactive Sliders
        card_y_start = y_offset
        card_height = int(370 * scale_y)
        self.draw_rounded_rect(panel, (margin_x, card_y_start),
                              (panel_width - margin_x, card_y_start + card_height),
                              (45, 45, 48), -1, int(12 * base_scale))

        y_offset += int(22 * scale_y)
        cv2.putText(panel, "SETTINGS", (margin_x + card_padding, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_section, (120, 200, 255), thickness_bold)
        y_offset += int(30 * scale_y)

        # Interactive sliders
        slider_x = margin_x + card_padding

        # Detection confidence slider
        self.draw_slider(panel, slider_x, y_offset, slider_width, slider_height,
                        self.config.detection_confidence, 0.0, 1.0,
                        "Detection Confidence", base_scale, "Detection")
        y_offset += line_spacing

        # Tracking confidence slider
        self.draw_slider(panel, slider_x, y_offset, slider_width, slider_height,
                        self.config.tracking_confidence, 0.0, 1.0,
                        "Tracking Confidence", base_scale, "Tracking")
        y_offset += line_spacing

        # Max hands slider
        self.draw_slider(panel, slider_x, y_offset, slider_width, slider_height,
                        self.config.max_hands, 1, 4,
                        "Max Hands", base_scale, "Max Hands", integer=True)
        y_offset += line_spacing

        # Processing scale slider
        self.draw_slider(panel, slider_x, y_offset, slider_width, slider_height,
                        self.config.processing_scale, 30, 100,
                        "Processing Scale", base_scale, "Scale", integer=True)
        y_offset += line_spacing

        # Draw thickness slider
        self.draw_slider(panel, slider_x, y_offset, slider_width, slider_height,
                        self.config.draw_thickness, 1, 5,
                        "Draw Thickness", base_scale, "Thickness", integer=True)
        y_offset += line_spacing

        # Toggle switches
        y_offset += int(10 * scale_y)
        toggle_spacing = int(32 * scale_y)

        # Face Mesh toggle
        self.draw_toggle(panel, slider_x, y_offset, 50, 24,
                        self.config.face_enabled, "Face Mesh Detection", base_scale, "Face Mesh")
        y_offset += toggle_spacing

        # Face Skeleton toggle
        self.draw_toggle(panel, slider_x, y_offset, 50, 24,
                        self.config.face_skeleton_visible, "Face Skeleton Visible", base_scale, "Face Skeleton")
        y_offset += toggle_spacing

        # Deepfake Card
        y_offset = card_y_start + card_height + int(15 * scale_y)
        card_y_start = y_offset
        card_height = int(130 * scale_y)
        self.draw_rounded_rect(panel, (margin_x, card_y_start),
                              (panel_width - margin_x, card_y_start + card_height),
                              (48, 42, 45), -1, int(12 * base_scale))

        y_offset += int(22 * scale_y)
        cv2.putText(panel, "DEEPFAKE", (margin_x + card_padding, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_section, (255, 150, 80), thickness_bold)
        y_offset += int(30 * scale_y)

        # Deepfake enable toggle
        self.draw_toggle(panel, slider_x, y_offset, 50, 24,
                        self.config.deepfake_enabled, "Deepfake Enabled", base_scale, "Deepfake")
        y_offset += toggle_spacing + int(5 * scale_y)

        # Deepfake mode slider
        self.draw_slider(panel, slider_x, y_offset, slider_width, slider_height,
                        self.config.deepfake_mode, 0, 2,
                        "Mode (0:Blur 1:Pixel 2:Swap)", base_scale, "DF Mode", integer=True)
        y_offset += line_spacing

        # Live Info Card
        y_offset = card_y_start + card_height + int(15 * scale_y)
        card_y_start = y_offset
        card_height = int(170 * scale_y)
        self.draw_rounded_rect(panel, (margin_x, card_y_start),
                              (panel_width - margin_x, card_y_start + card_height),
                              (42, 48, 45), -1, int(12 * base_scale))

        y_offset += int(22 * scale_y)
        cv2.putText(panel, "LIVE INFO", (margin_x + card_padding, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_section, (100, 255, 180), thickness_bold)
        y_offset += int(28 * scale_y)

        info_items = [
            ("FPS", f"{info_text.get('fps', 0):.1f}", (150, 200, 255)),
            ("Total Fingers", str(info_text.get('fingers', 0)), (255, 200, 100)),
            ("Hands Detected", str(info_text.get('hands', 0)), (150, 255, 150)),
            ("Faces Detected", str(info_text.get('faces', 0)), (255, 150, 200)),
        ]

        for label, value, color in info_items:
            cv2.putText(panel, label, (margin_x + card_padding, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (200, 200, 200), thickness_normal)

            # Value with colored background
            value_x = panel_width - margin_x - card_padding - 60
            cv2.rectangle(panel, (value_x - 5, y_offset - 15),
                         (value_x + 50, y_offset + 5), (30, 30, 35), -1)
            cv2.putText(panel, value, (value_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, color, thickness_bold)
            y_offset += int(32 * scale_y)

        # Keyboard Shortcuts Card
        y_offset = card_y_start + card_height + int(15 * scale_y)
        card_y_start = y_offset
        card_height = int(130 * scale_y)
        self.draw_rounded_rect(panel, (margin_x, card_y_start),
                              (panel_width - margin_x, card_y_start + card_height),
                              (40, 45, 50), -1, int(12 * base_scale))

        y_offset += int(22 * scale_y)
        cv2.putText(panel, "SHORTCUTS", (margin_x + card_padding, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_section, (255, 220, 100), thickness_bold)
        y_offset += int(28 * scale_y)

        shortcuts = [
            ("Q", "Quit Application"),
            ("S", "Save Screenshot"),
            ("R", "Reset Stored Face")
        ]

        for key, desc in shortcuts:
            # Draw key badge
            key_x = margin_x + card_padding
            self.draw_rounded_rect(panel, (key_x, y_offset - 15),
                                  (key_x + 25, y_offset + 5),
                                  (70, 70, 80), -1, int(5 * base_scale))
            cv2.putText(panel, key, (key_x + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (255, 255, 255), thickness_bold)

            # Description
            cv2.putText(panel, desc, (key_x + 35, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (180, 180, 180), thickness_normal)
            y_offset += int(28 * scale_y)

        return panel

    def show_frame_and_panel(self, frame: np.ndarray, info: dict):
        """
        Display video frame and control panel in separate windows.
        
        Args:
            frame: Video frame
            info: Info dictionary for panel
        """
        # Resize frame to target resolution if needed
        if frame.shape[:2] != (self.video_height, self.video_width):
            frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        # Get actual control window size (in case user resized it)
        try:
            # Try to get window rect (not all OpenCV builds support this)
            rect = cv2.getWindowImageRect(self.control_window)
            if rect[2] > 0 and rect[3] > 0:
                actual_width = rect[2]
                actual_height = rect[3]
            else:
                actual_width = self.control_width
                actual_height = self.control_height
        except:
            # Fallback to default size
            actual_width = self.control_width
            actual_height = self.control_height
        
        # Create panel with actual size
        panel = self.create_control_panel(info, actual_width, actual_height)
        
        # Display in separate windows
        cv2.imshow(self.video_window, frame)
        cv2.imshow(self.control_window, panel)


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

    def draw_face(self, frame, face_results, draw_thickness: int, draw_skeleton: bool = True) -> int:
        """
        Draw face mesh on frame.

        Args:
            frame: Image to draw on
            face_results: MediaPipe face detection results
            draw_thickness: Thickness of drawing lines
            draw_skeleton: Whether to draw the face skeleton/mesh

        Returns:
            Number of faces detected
        """
        num_faces = 0

        if not face_results or not face_results.multi_face_landmarks:
            return num_faces

        num_faces = len(face_results.multi_face_landmarks)

        if draw_skeleton:
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

        # Set 1920x1080 resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")

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
                    frame, face_results, config.draw_thickness, config.face_skeleton_visible
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

                # Display in separate windows
                self.ui.show_frame_and_panel(frame, info)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(frame)
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