# Hand & Face Tracking Application

A real-time computer vision application that detects and tracks hand movements and facial features using your webcam. Built with MediaPipe and OpenCV for CPU-friendly performance.

## Features

- **Real-time Hand Tracking** - Detect and track up to 2 hands simultaneously with 21 landmarks per hand
- **Intelligent Finger Counting** - Automatically counts raised fingers for each hand with visual feedback
- **Hand Skeleton Visualization** - Displays hand joints and connections in real-time
- **Face Mesh Tracking** - Optional face detection with 468 facial landmarks and tessellation
- **Interactive Controls** - Adjust detection parameters in real-time via trackbars
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
1. Open your webcam at 1280x720 resolution
2. Display a "Controls" window with adjustable parameters
3. Show the video feed with hand/face tracking overlays

### Keyboard Controls

- **q** - Quit the application

## Interactive Controls

The application provides 6 real-time adjustable controls:

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| **Det%** | 0-100 | 50 | Detection confidence threshold (higher = more strict detection) |
| **Track%** | 0-100 | 50 | Tracking confidence threshold (higher = more stable tracking) |
| **Hands** | 1-2 | 2 | Maximum number of hands to detect simultaneously |
| **Face** | 0-1 | 1 | Enable (1) or disable (0) face mesh tracking |
| **Scale%** | 30-100 | 100 | Processing resolution scale (lower = faster but less accurate) |
| **Thick** | 1-4 | 2 | Drawing thickness for skeleton lines and landmarks |

## How It Works

1. **Capture** - Reads frames from webcam in mirror mode (horizontally flipped)
2. **Scale** - Optionally downscales frames for faster processing based on Scale% setting
3. **Detect** - Uses MediaPipe to detect hands and optionally faces
4. **Analyze** - Counts raised fingers using heuristic detection (compares finger tip positions with PIP joints)
5. **Visualize** - Draws hand skeleton (green joints, white connections), face mesh (orange), finger counts, and FPS
6. **Display** - Shows annotated video feed in real-time

### Finger Counting Logic

- Detects if finger tips are above proximal interphalangeal (PIP) joints
- Special thumb detection based on hand handedness (left/right)
- Displays individual hand counts and total finger count

## Technical Details

- **Resolution**: 1280x720 default capture resolution
- **Model Complexity**: 0 (CPU-friendly lightweight models)
- **Hand Landmarks**: 21 points per hand (wrist, palm, finger joints)
- **Face Landmarks**: 468 points with face mesh tessellation
- **Performance**: Adaptive FPS calculation with exponential smoothing

## Visualization

- **Hand Joints**: Green circles
- **Hand Connections**: White lines
- **Face Mesh**: Orange landmarks and tessellation
- **Labels**: Hand type (Left/Right) with finger count
- **HUD**: Total fingers count and FPS in top-left corner

## Project Status

Currently on branch `feat/face-tracking` with recent additions:
- Face tracking feature
- Finger count overlay
- Thicker skeleton visualization
- FPS monitoring

---

*Built with MediaPipe and OpenCV*
