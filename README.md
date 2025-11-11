# Hand & Face Tracking Application

A real-time computer vision application that detects and tracks hand movements and facial features using your webcam. Built with MediaPipe and OpenCV for CPU-friendly performance.

## Features

- **Real-time Hand Tracking** - Detect and track up to 4 hands simultaneously with 21 landmarks per hand
- **Intelligent Finger Counting** - Automatically counts raised fingers for each hand with visual feedback
- **Hand Skeleton Visualization** - Displays hand joints and connections in real-time
- **Face Mesh Tracking** - Optional face detection with 468 facial landmarks and tessellation
- **Face Skeleton Toggle** - Show or hide face mesh while keeping detection active
- **Deepfake Effects** - Real-time face manipulation with blur, pixelation, and face swap modes
- **Modern Interactive UI** - Beautiful card-based control dashboard with integrated sliders and toggles
- **Unified Controls** - All settings in one cohesive interface with smooth, responsive controls
- **Performance Monitoring** - Live FPS counter to monitor application performance
- **CPU Optimized** - Lightweight model configuration for smooth performance on standard hardware

## Requirements

- Python 3.7+
- Webcam

### Dependencies

- MediaPipe 0.10.21
- OpenCV-Python 4.10.0.84
- NumPy 1.26.4

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Hand_Tracking
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

The application will:
1. Open your webcam at 1920x1080 resolution
2. Display a modern "Control Dashboard" with interactive sliders and toggles
3. Show the video feed with hand/face tracking overlays

### Keyboard Controls

- **Q** - Quit the application
- **S** - Save screenshot of current frame
- **R** - Reset stored face (for face swap mode)

## Interactive Controls

The unified control dashboard features interactive sliders and toggles organized into themed cards:

### Settings Card

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **Detection Confidence** | 0-100% | 50% | Detection confidence threshold (higher = more strict detection) |
| **Tracking Confidence** | 0-100% | 50% | Tracking confidence threshold (higher = more stable tracking) |
| **Max Hands** | 1-4 | 2 | Maximum number of hands to detect simultaneously |
| **Processing Scale** | 30-100% | 60% | Processing resolution scale (lower = faster but less accurate) |
| **Draw Thickness** | 1-5 | 2 | Drawing thickness for skeleton lines and landmarks |
| **Face Mesh Detection** | ON/OFF | ON | Enable or disable face mesh tracking |
| **Face Skeleton Visible** | ON/OFF | ON | Show or hide face skeleton (detection remains active) |

### Deepfake Card

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **Deepfake Enabled** | ON/OFF | OFF | Enable or disable face manipulation effects |
| **Mode** | 0-2 | 0 | Effect mode: 0=Blur, 1=Pixelate, 2=Face Swap |

### Live Info Card

Real-time statistics display:
- **FPS** - Current frames per second
- **Total Fingers** - Count of all raised fingers across detected hands
- **Hands Detected** - Number of hands currently in frame
- **Faces Detected** - Number of faces currently in frame

## How It Works

1. **Capture** - Reads frames from webcam in mirror mode (horizontally flipped)
2. **Scale** - Optionally downscales frames for faster processing based on Processing Scale setting
3. **Detect** - Uses MediaPipe to detect hands and optionally faces
4. **Process** - Applies deepfake effects (blur, pixelate, or face swap) if enabled
5. **Analyze** - Counts raised fingers using heuristic detection (compares finger tip positions with PIP joints)
6. **Visualize** - Draws hand skeleton (green joints, white connections), optionally face mesh (blue), finger counts, and FPS
7. **Display** - Shows annotated video feed and interactive control dashboard in real-time

### Finger Counting Logic

- Detects if finger tips are above proximal interphalangeal (PIP) joints
- Special thumb detection based on hand handedness (left/right)
- Displays individual hand counts and total finger count

### Deepfake Effects

- **Blur Mode** - Applies Gaussian blur to detected face regions for privacy
- **Pixelate Mode** - Creates pixelation effect on faces for anonymization
- **Face Swap Mode** - Stores first detected face and swaps it with subsequent faces (experimental)

## Technical Details

- **Resolution**: 1920x1080 default capture resolution
- **Model Complexity**: 0 (CPU-friendly lightweight models)
- **Hand Landmarks**: 21 points per hand (wrist, palm, finger joints)
- **Face Landmarks**: 468 points with face mesh tessellation
- **Performance**: Adaptive FPS calculation with exponential smoothing
- **UI Framework**: Custom OpenCV-based interactive controls with mouse callbacks
- **Control System**: Unified dashboard with card-based layout and real-time updates

## Visualization

### Video Feed
- **Hand Joints**: Green circles
- **Hand Connections**: White lines
- **Face Mesh**: Blue landmarks and tessellation (toggleable)
- **Labels**: Hand type (Left/Right) with finger count
- **Effects**: Optional blur, pixelation, or face swap

### Control Dashboard
- **Modern Card UI**: Rounded cards with gradient background
- **Interactive Sliders**: Draggable handles with real-time value display
- **Toggle Switches**: iOS-style switches with visual feedback
- **Live Statistics**: Color-coded metrics with background highlights
- **Keyboard Shortcuts**: Badge-style key indicators with descriptions

## Code Architecture

The application follows Object-Oriented Programming principles with a modular design:

### Core Classes

- **`AppConfig`** - Dataclass holding all application configuration settings
- **`UnifiedUI`** - Manages modern card-based UI with interactive sliders and toggles
- **`MediaPipeDetector`** - Handles MediaPipe hand and face detection models
- **`DeepfakeProcessor`** - Processes face manipulation effects (blur, pixelate, swap)
- **`HandGestureAnalyzer`** - Analyzes hand landmarks and counts fingers
- **`Renderer`** - Responsible for all drawing and visualization operations
- **`HandFaceTracker`** - Main application orchestrator, manages the tracking loop

### UI System

- **Custom Sliders**: Interactive draggable sliders with visual feedback
- **Toggle Switches**: iOS-style toggle buttons for boolean settings
- **Mouse Callbacks**: Event-driven interaction for real-time control updates
- **Card-Based Layout**: Organized sections with rounded corners and gradient backgrounds
- **Responsive Design**: Scales with window resizing

### Benefits

- **Modularity**: Each class has a single, well-defined responsibility
- **Maintainability**: Easy to modify or extend individual components
- **Reusability**: Components can be imported and used in other projects
- **Testability**: Classes can be unit tested independently
- **Clean Code**: No global variables, proper encapsulation, and clear interfaces
- **Modern UX**: Professional, intuitive control interface

## Project Status

**Latest Update: Modern UI Redesign & Deepfake Features**
- Complete UI redesign with card-based layout and gradient backgrounds
- Integrated interactive sliders and toggle switches directly into dashboard
- Added deepfake effects: blur, pixelate, and face swap modes
- Face skeleton visibility toggle (keeps detection active for effects)
- Unified control interface with mouse-driven interactions
- Beautiful, responsive design with rounded corners and visual feedback
- Increased resolution support (1920x1080)

**Previous Updates:**
- Complete OOP refactoring
- 5+ well-defined classes with clear responsibilities
- Improved code organization and maintainability
- Added comprehensive docstrings and type hints

**Earlier Updates:**
- Face tracking feature
- Finger count overlay
- Thicker skeleton visualization
- FPS monitoring

---

*Built with MediaPipe and OpenCV*
