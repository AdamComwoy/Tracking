import time
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Hrubšie kreslenie kostry
LANDMARK_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
CONNECTION_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


def count_raised_fingers(landmarks, handed_label: str) -> int:
    """
    landmarks: MediaPipe landmarks (21 bodov), normalizované .x/.y (0..1)
    handed_label: 'Left' alebo 'Right' (z MediaPipe handedness)
    Heuristika:
      - index, middle, ring, pinky: TIP.y < PIP.y
      - thumb: pre 'Right' TIP.x > IP.x, pre 'Left' TIP.x < IP.x
    """
    if not landmarks or len(landmarks) != 21:
        return 0

    up = 0

    # 4 prsty (okrem palca)
    for tip in [8, 12, 16, 20]:
        pip_idx = tip - 2
        if landmarks[tip].y < landmarks[pip_idx].y:
            up += 1

    # palec (TIP 4 vs IP 3)
    tip = 4
    ip = 3
    if handed_label == "Right":
        if landmarks[tip].x > landmarks[ip].x:
            up += 1
    else:  # 'Left'
        if landmarks[tip].x < landmarks[ip].x:
            up += 1

    return up


def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"⚠️  Kamera sa nedá otvoriť (index {camera_index}).")
        return

    # voliteľne nižšie rozlíšenie pre plynulejší beh
    # cap.set(3, 640); cap.set(4, 480)

    prev_time = time.time()
    fps = 0.0

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  Nedá sa čítať snímok z kamery.")
                break

            frame = cv2.flip(frame, 1)  # selfie
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            total = 0

            # Kresli kostru a per-hand počítanie
            if results.multi_hand_landmarks:
                for landmarks, handed in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handed.classification[0].label  # 'Left' / 'Right'
                    count = count_raised_fingers(landmarks.landmark, label)
                    total += count

                    # kostra – hrubšie čiary
                    mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        LANDMARK_STYLE,
                        CONNECTION_STYLE,
                    )

                    # vypočítaj bounding box ~ z landmarkov, aby sme vedeli kam dať text
                    h, w = frame.shape[:2]
                    xs = [int(p.x * w) for p in landmarks.landmark]
                    ys = [int(p.y * h) for p in landmarks.landmark]
                    x, y = max(min(xs), 0), max(min(ys) - 10, 20)

                    put_text(frame, f"{label}: {count}", (x, y), scale=0.8, color=(0, 255, 255))

            # FPS (vyhladené jednoduchým priemerom)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            put_text(frame, f"Total: {total}", (12, 36), scale=1.0)
            put_text(frame, f"FPS: {fps:.1f}", (12, 68), scale=0.8, color=(0, 200, 255))

            cv2.imshow("Hand tracking - skeleton + per-hand finger count (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
