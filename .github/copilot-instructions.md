### Project quick facts
- Language: Python 3.x
- Purpose: Real-time hand-tracking demo using MediaPipe + OpenCV (single-file app at `main.py`).
- Entrypoint: `main.py` (call `python main.py [camera_index]`).
- Dependencies: declared in `requirements.txt` (MediaPipe, opencv-python, numpy).

### What this repository expects from an AI code contributor
- Keep changes minimal and focused: this is a single-purpose demo app. Avoid introducing large frameworks or heavy refactors.
- Preserve explicit hardware handling: `main.py` opens the camera with the DirectShow flag (`cv2.VideoCapture(..., cv2.CAP_DSHOW)`) — keep or document platform-specific choices.

### Useful patterns & examples (from the code)
- Camera capture and mirror: frame is flipped horizontally with `cv2.flip(frame, 1)` before processing and display.
- Performance & mutability: input to MediaPipe is converted to RGB and `rgb.flags.writeable` is toggled to avoid unnecessary copies when calling `hands.process(rgb)`.
- Drawing: landmarks are rendered with `mp.solutions.drawing_utils` and `mp.solutions.drawing_styles`.

### How to run locally (Windows notes)
1. Create a virtual env (recommended): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `python main.py` or `python main.py 1` (pass camera index if needed). Quit with `q` in the window.

### Project-specific conventions and gotchas
- Single-file demo: No package layout — keep imports and logic in `main.py` unless adding tests or a small module split that remains clearly documented.
- Camera index behavior: The app expects an integer index; validate user input when adding CLI features.
- Internationalized messages: `main.py` prints some Slovak messages — be conservative modifying user-facing text.
- GUI loop: uses `cv2.imshow` + `cv2.waitKey(1)`; avoid blocking waits >1 in demos.

### Integration points and external dependencies
- MediaPipe Hands: configured in `with mp_hands.Hands(...):` — prefer adjusting `model_complexity`, `max_num_hands`, and detection/tracking confidence via small changes only.
- OpenCV VideoCapture: uses DirectShow (`cv2.CAP_DSHOW`) on Windows — when adding cross-platform support, guard this flag behind an OS check.

### When making edits, the AI should:
- Update `requirements.txt` if new packages are introduced.
- Add minimal, executable examples or tests if adding public-facing functions.
- Keep prints and UI behavior explicit; don't change the default camera index logic without adding a clear CLI.

### Files to reference when coding
- `main.py` — primary logic
- `requirements.txt` — dependency versions

If anything here is unclear or you want more detailed guidance (for example: adding a CLI, packaging the demo, or adding unit tests), tell me which direction and I'll extend these instructions.
