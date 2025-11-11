import time
import cv2
import mediapipe as mp

# --- MediaPipe moduly ---
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
#awdoanwdawod 
# ---------- UI: trackbary ----------
def on_trackbar(_=None):
    pass

def create_controls():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 520, 280)
    # krátke názvy, aby OpenCV nestrihalo text
    cv2.createTrackbar("Det%",   "Controls", 50, 100, on_trackbar)  # 0..1
    cv2.createTrackbar("Track%", "Controls", 50, 100, on_trackbar)  # 0..1
    cv2.createTrackbar("Hands",  "Controls", 2,    2,  on_trackbar) # 1..2
    cv2.createTrackbar("Face",   "Controls", 1,    1,  on_trackbar) # 0/1
    cv2.createTrackbar("Scale%", "Controls", 60, 100, on_trackbar)  # 30..100
    cv2.createTrackbar("Thick",  "Controls", 2,    4,  on_trackbar) # 1..4

def get_trackbar_values():
    det_c = cv2.getTrackbarPos("Det%", "Controls") / 100.0
    trk_c = cv2.getTrackbarPos("Track%", "Controls") / 100.0
    max_hands = max(1, cv2.getTrackbarPos("Hands", "Controls"))
    face_on = cv2.getTrackbarPos("Face", "Controls")
    proc_scale = max(30, cv2.getTrackbarPos("Scale%", "Controls"))
    draw_th = max(1, cv2.getTrackbarPos("Thick", "Controls"))
    return det_c, trk_c, max_hands, face_on, proc_scale, draw_th

# ---------- Pomocné ----------
def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def count_fingers(lm, label):
    """Heuristika: TIP.y < PIP.y pre 4 prsty; palec podľa handedness (TIP x vs IP x)."""
    if not lm or len(lm) != 21:
        return 0
    up = 0
    for tip in [8, 12, 16, 20]:  # index, middle, ring, pinky
        if lm[tip].y < lm[tip - 2].y:
            up += 1
    # palec (4 vs 3)
    if label == "Right" and lm[4].x > lm[3].x:
        up += 1
    if label == "Left" and lm[4].x < lm[3].x:
        up += 1
    return up

# ---------- Hlavný program ----------
def main(camera_index: int = 0):
    """Demo: ruky (povinne) + voliteľne FaceMesh, všetko na CPU (model_complexity=0)."""
    create_controls()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f" Kamera sa nedá otvoriť (index {camera_index}).")
        return

    # vyššie rozlíšenie náhľadu; spracovanie riadi Scale%
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = None
    face = None
    prev_params = None
    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Nedá sa čítať snímok z kamery.")
            break

        frame = cv2.flip(frame, 1)  # selfie
        H, W = frame.shape[:2]

        # UI hodnoty
        det_c, trk_c, max_hands, face_on, proc_scale, draw_th = get_trackbar_values()

        # (Re)inicializácia MediaPipe pri zmene parametrov
        params = (det_c, trk_c, max_hands, face_on)
        if params != prev_params:
            if hands:
                hands.close()
            if face:
                face.close()

            hands = mp_hands.Hands(
                model_complexity=0,               # CPU-friendly model
                max_num_hands=max_hands,
                min_detection_confidence=det_c,
                min_tracking_confidence=trk_c,
            )
            face = (
                mp_face.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=det_c,
                    min_tracking_confidence=trk_c,
                )
                if face_on
                else None
            )
            prev_params = params

        # Downscale na spracovanie (Scale %)
        scale = proc_scale / 100.0
        if scale < 1.0:
            proc = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc = frame

        # MediaPipe chce RGB
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hres = hands.process(rgb) if hands else None
        fres = face.process(rgb) if face else None
        rgb.flags.writeable = True

        # Kreslenie rúk + počty prstov
        total = 0
        if hres and hres.multi_hand_landmarks:
            for lm, handed in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                label = handed.classification[0].label  # 'Left' / 'Right'
                count = count_fingers(lm.landmark, label)
                total += count

                mp_draw.draw_landmarks(
                    frame,
                    lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=draw_th, circle_radius=max(1, draw_th - 1)),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=draw_th),
                )

                xs = [int(p.x * W) for p in lm.landmark]
                ys = [int(p.y * H) for p in lm.landmark]
                x, y = max(min(xs), 10), max(min(ys) - 10, 20)
                put_text(frame, f"{label}: {count}", (x, y), scale=0.7, color=(0, 255, 255), thick=2)

        # Kreslenie Face Mesh (voliteľné)
        if fres and fres.multi_face_landmarks:
            for face_lm in fres.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_lm,
                    mp_face.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(0, 150, 255), thickness=max(1, draw_th - 1), circle_radius=1),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=max(1, draw_th - 1)),
                )

        # FPS
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        put_text(frame, f"Total: {total}", (12, 36), scale=1.0, color=(255, 255, 255), thick=2)
        put_text(frame, f"FPS: {fps:.1f}", (12, 68), scale=0.8, color=(0, 200, 255), thick=2)

        cv2.imshow("Ruky + (voliteľne) Face  —  stlač 'q' pre ukončenie", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if hands:
        hands.close()
    if face:
        face.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
