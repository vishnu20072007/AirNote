import cv2
import mediapipe as mp
import time
import math

# ============================================================
# AIRNOTE - STRICT ERASER TEST V2
# Proper Fist -> ERASER
# ============================================================

# ---------------- MEDIAPIPE ----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="vision/models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,

    min_hand_detection_confidence=0.70,
    min_hand_presence_confidence=0.70,
    min_tracking_confidence=0.70,
)


# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()

camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# ERASER DETECTION SETTINGS
# ============================================================

# All four main fingers must be folded.
REQUIRED_FOLDED_FINGERS = 4

# Fist must remain detected for these many frames.
CONFIRM_FRAMES = 5

eraser_frames = 0
eraser_mode = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def finger_fold_ratio(hand, tip_id, pip_id):
    """
    Compares fingertip distance from wrist
    with PIP joint distance from wrist.

    Smaller ratio = finger is more folded.
    """

    wrist = hand[0]

    tip = hand[tip_id]
    pip = hand[pip_id]

    tip_distance = distance(tip, wrist)
    pip_distance = distance(pip, wrist)

    if pip_distance == 0:
        return 999

    return tip_distance / pip_distance


def is_fist(hand):
    """
    Strict fist detection.

    Index, middle, ring and pinky must all
    be clearly folded.
    """

    index_ratio = finger_fold_ratio(
        hand,
        8,
        6
    )

    middle_ratio = finger_fold_ratio(
        hand,
        12,
        10
    )

    ring_ratio = finger_fold_ratio(
        hand,
        16,
        14
    )

    pinky_ratio = finger_fold_ratio(
        hand,
        20,
        18
    )


    # Lower ratio means fingertip is closer
    # to wrist than its PIP joint.
    #
    # This is a stricter condition than
    # simply checking 3 out of 4 fingers.

    index_folded = index_ratio < 1.0
    middle_folded = middle_ratio < 1.0
    ring_folded = ring_ratio < 1.0
    pinky_folded = pinky_ratio < 1.0


    folded_count = sum([
        index_folded,
        middle_folded,
        ring_folded,
        pinky_folded
    ])


    return folded_count == REQUIRED_FOLDED_FINGERS


# ============================================================
# FPS
# ============================================================

fps_start = time.time()
fps_counter = 0
fps = 0


# ============================================================
# HAND TRACKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame")
            break

        frame = cv2.flip(frame, 1)


        # ----------------------------------------------------
        # MEDIAPIPE INPUT
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            time.time() * 1000
        )

        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        status = "NO HAND"


        # ====================================================
        # HAND FOUND
        # ====================================================

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]


            # ------------------------------------------------
            # STRICT FIST CHECK
            # ------------------------------------------------

            fist_detected = is_fist(hand)


            # ------------------------------------------------
            # CONFIRM FIST
            # ------------------------------------------------

            if fist_detected:

                eraser_frames += 1

            else:

                eraser_frames = 0


            # ------------------------------------------------
            # ENTER ERASER MODE
            # ------------------------------------------------

            if (
                not eraser_mode
                and eraser_frames >= CONFIRM_FRAMES
            ):

                eraser_mode = True


            # ------------------------------------------------
            # EXIT ERASER MODE
            # ------------------------------------------------

            if (
                eraser_mode
                and not fist_detected
            ):

                eraser_mode = False
                eraser_frames = 0


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if eraser_mode:

                status = "ERASER"

            elif fist_detected:

                status = "CONFIRMING..."

            else:

                status = "IDLE"


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for landmark in hand:

                x = int(
                    landmark.x * frame_width
                )

                y = int(
                    landmark.y * frame_height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            eraser_mode = False
            eraser_frames = 0

            status = "NO HAND"


        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1.0:

            fps = fps_counter / elapsed

            fps_counter = 0
            fps_start = time.time()


        # ====================================================
        # UI
        # ====================================================

        cv2.putText(
            frame,
            status,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "Close all four fingers = Eraser",
            (20, frame_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AirNote - Strict Eraser Test V2",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("AirNote Strict Eraser Test finished.")