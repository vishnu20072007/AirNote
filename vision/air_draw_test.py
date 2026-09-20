import cv2
import mediapipe as mp
import time
import math
from collections import deque

# ============================================================
# AIRNOTE - PRECISION V4
# Stable Cursor + Low Latency + Smooth Drawing
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
# DRAWING
# ============================================================

canvas = None

BRUSH_SIZE = 4

# Tiny movement ignored
DEAD_ZONE = 1.8

# Small history only -> less lag
HISTORY_SIZE = 3

position_history = deque(maxlen=HISTORY_SIZE)


# ============================================================
# PINCH
# ============================================================

PINCH_START = 0.31
PINCH_STOP = 0.40

PINCH_CONFIRM_FRAMES = 3

pinch_frames = 0
drawing = False


# ============================================================
# STABLE CURSOR FILTER
# ============================================================

cursor_x = None
cursor_y = None

previous_x = None
previous_y = None


# ============================================================
# FILTER SETTINGS
# ============================================================

# Slow movement -> stable
SLOW_SMOOTHING = 0.35

# Normal movement -> balanced
NORMAL_SMOOTHING = 0.72

# Fast movement -> almost immediate
FAST_SMOOTHING = 0.94


# ============================================================
# FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def median_position(history):

    if not history:
        return None

    xs = sorted(p[0] for p in history)
    ys = sorted(p[1] for p in history)

    middle = len(xs) // 2

    return xs[middle], ys[middle]


def reset_cursor():

    global cursor_x
    global cursor_y
    global previous_x
    global previous_y

    cursor_x = None
    cursor_y = None

    previous_x = None
    previous_y = None

    position_history.clear()


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
        # CAMERA FRAME
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = frame.copy()
            canvas[:] = 255


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)

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

            wrist = hand[0]
            thumb_tip = hand[4]
            index_tip = hand[8]
            middle_mcp = hand[9]


            # ------------------------------------------------
            # PINCH RATIO
            # ------------------------------------------------

            pinch_distance = distance(
                thumb_tip,
                index_tip
            )

            palm_size = distance(
                wrist,
                middle_mcp
            )

            if palm_size > 0:

                pinch_ratio = (
                    pinch_distance / palm_size
                )

            else:

                pinch_ratio = 999


            # ------------------------------------------------
            # PINCH CONFIRMATION
            # ------------------------------------------------

            if pinch_ratio < PINCH_START:

                pinch_frames += 1

            else:

                pinch_frames = 0


            # ------------------------------------------------
            # START DRAWING
            # ------------------------------------------------

            if (
                not drawing
                and pinch_frames >= PINCH_CONFIRM_FRAMES
            ):

                drawing = True

                reset_cursor()


            # ------------------------------------------------
            # STOP DRAWING
            # ------------------------------------------------

            if drawing and pinch_ratio > PINCH_STOP:

                drawing = False

                reset_cursor()

                pinch_frames = 0


            # =================================================
            # DRAWING
            # =================================================

            if drawing:

                status = "DRAWING"


                # ---------------------------------------------
                # INDEX TIP = CURSOR
                # ---------------------------------------------

                raw_x = index_tip.x * frame_width
                raw_y = index_tip.y * frame_height


                # ---------------------------------------------
                # FILTER HISTORY
                # ---------------------------------------------

                position_history.append(
                    (raw_x, raw_y)
                )

                target = median_position(
                    position_history
                )


                if target is not None:

                    target_x, target_y = target


                    # -----------------------------------------
                    # FIRST POSITION
                    # -----------------------------------------

                    if cursor_x is None:

                        cursor_x = target_x
                        cursor_y = target_y

                        previous_x = cursor_x
                        previous_y = cursor_y


                    else:

                        movement = math.hypot(
                            target_x - cursor_x,
                            target_y - cursor_y
                        )


                        # -------------------------------------
                        # DEAD ZONE
                        # -------------------------------------

                        if movement < DEAD_ZONE:

                            target_x = cursor_x
                            target_y = cursor_y


                        # -------------------------------------
                        # ADAPTIVE RESPONSE
                        # -------------------------------------

                        if movement < 4:

                            # Very small movement:
                            # maximum stability
                            smoothing = SLOW_SMOOTHING

                        elif movement < 15:

                            # Normal handwriting
                            smoothing = NORMAL_SMOOTHING

                        else:

                            # Fast movement:
                            # minimum latency
                            smoothing = FAST_SMOOTHING


                        # -------------------------------------
                        # UPDATE CURSOR
                        # -------------------------------------

                        cursor_x += (
                            target_x - cursor_x
                        ) * smoothing

                        cursor_y += (
                            target_y - cursor_y
                        ) * smoothing


                    current_x = cursor_x
                    current_y = cursor_y


                    # -----------------------------------------
                    # DRAWING LINE
                    # -----------------------------------------

                    if (
                        previous_x is not None
                        and previous_y is not None
                    ):

                        movement = math.hypot(
                            current_x - previous_x,
                            current_y - previous_y
                        )

                        if movement >= DEAD_ZONE:

                            cv2.line(
                                canvas,

                                (
                                    int(previous_x),
                                    int(previous_y)
                                ),

                                (
                                    int(current_x),
                                    int(current_y)
                                ),

                                (0, 0, 0),

                                BRUSH_SIZE,

                                cv2.LINE_AA
                            )


                    previous_x = current_x
                    previous_y = current_y


                    # -----------------------------------------
                    # STABLE GREEN DOT
                    # -----------------------------------------

                    cv2.circle(
                        frame,

                        (
                            int(current_x),
                            int(current_y)
                        ),

                        7,

                        (0, 255, 0),

                        -1,

                        cv2.LINE_AA
                    )


            else:

                status = "PINCH TO DRAW"

                reset_cursor()


            # =================================================
            # THUMB + INDEX MARKERS
            # =================================================

            thumb_x = int(
                thumb_tip.x * frame_width
            )

            thumb_y = int(
                thumb_tip.y * frame_height
            )

            index_x = int(
                index_tip.x * frame_width
            )

            index_y = int(
                index_tip.y * frame_height
            )


            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                5,
                (255, 0, 0),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (index_x, index_y),
                5,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            drawing = False
            pinch_frames = 0

            reset_cursor()

            status = "NO HAND"


        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1:

            fps = fps_counter / elapsed

            fps_counter = 0
            fps_start = time.time()


        # ====================================================
        # OUTPUT
        # ====================================================

        output = cv2.addWeighted(
            frame,
            0.72,
            canvas,
            0.28,
            0
        )


        cv2.putText(
            output,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            output,
            f"FPS: {fps:.0f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            output,
            "Pinch Thumb + Index = Draw",
            (20, frame_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.imshow(
            "AirNote - Precision V4",
            output
        )


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("AirNote Precision V4 finished.")