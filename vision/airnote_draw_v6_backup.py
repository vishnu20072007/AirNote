import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# ============================================================
# AIRNOTE - SMOOTH DRAW V6
# Edge Lock + Stable Stroke + Pinch Draw
# ============================================================

MODEL_PATH = "vision/models/hand_landmarker.task"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480


# ============================================================
# DRAW SETTINGS
# ============================================================

BRUSH_SIZE = 3

MIN_MOVEMENT = 1.5
MAX_JUMP = 70


# ============================================================
# SMOOTHING
# ============================================================

SMOOTHING = 0.40

# Small history removes single-frame fingertip noise
HISTORY_SIZE = 3


# ============================================================
# PINCH SETTINGS
# ============================================================

TOUCH_START = 0.28
TOUCH_STOP = 0.36


# ============================================================
# EDGE LOCK SETTINGS
# ============================================================

# When fingertip enters these zones, cursor locks to edge.

EDGE_START = 0.045
EDGE_RELEASE = 0.075


# ============================================================
# HAND LOSS
# ============================================================

MAX_LOST_FRAMES = 6


# ============================================================
# VARIABLES
# ============================================================

filtered_x = None
filtered_y = None

previous_point = None

drawing = False

lost_frames = 0

x_history = deque(maxlen=HISTORY_SIZE)
y_history = deque(maxlen=HISTORY_SIZE)

locked_left = False
locked_right = False
locked_top = False
locked_bottom = False


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.75,

    min_hand_presence_confidence=0.75,

    min_tracking_confidence=0.75
)


# ============================================================
# CANVAS
# ============================================================

canvas = np.full(
    (
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        3
    ),
    255,
    dtype=np.uint8
)


# ============================================================
# MAIN
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            break


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )

        frame = cv2.resize(
            frame,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            )
        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        timestamp = int(
            cv2.getTickCount()
            * 1000
            /
            cv2.getTickFrequency()
        )


        result = landmarker.detect_for_video(
            image,
            timestamp
        )


        # ====================================================
        # DISPLAY CANVAS
        # ====================================================

        display_canvas = canvas.copy()


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            lost_frames = 0

            hand = result.hand_landmarks[0]

            index = hand[8]

            thumb = hand[4]


            # =================================================
            # RAW INDEX
            # =================================================

            raw_x = clamp(
                index.x,
                0.0,
                1.0
            )

            raw_y = clamp(
                index.y,
                0.0,
                1.0
            )


            # =================================================
            # MEDIAN HISTORY
            # =================================================

            x_history.append(raw_x)
            y_history.append(raw_y)

            stable_x = float(
                np.median(
                    list(x_history)
                )
            )

            stable_y = float(
                np.median(
                    list(y_history)
                )
            )


            # =================================================
            # EDGE LOCK - LEFT
            # =================================================

            if locked_left:

                if stable_x > EDGE_RELEASE:

                    locked_left = False

            else:

                if stable_x < EDGE_START:

                    locked_left = True


            # =================================================
            # EDGE LOCK - RIGHT
            # =================================================

            if locked_right:

                if stable_x < (
                    1.0 - EDGE_RELEASE
                ):

                    locked_right = False

            else:

                if stable_x > (
                    1.0 - EDGE_START
                ):

                    locked_right = True


            # =================================================
            # EDGE LOCK - TOP
            # =================================================

            if locked_top:

                if stable_y > EDGE_RELEASE:

                    locked_top = False

            else:

                if stable_y < EDGE_START:

                    locked_top = True


            # =================================================
            # EDGE LOCK - BOTTOM
            # =================================================

            if locked_bottom:

                if stable_y < (
                    1.0 - EDGE_RELEASE
                ):

                    locked_bottom = False

            else:

                if stable_y > (
                    1.0 - EDGE_START
                ):

                    locked_bottom = True


            # =================================================
            # TARGET POSITION
            # =================================================

            target_x = stable_x
            target_y = stable_y


            # =================================================
            # APPLY EDGE LOCK
            # =================================================

            if locked_left:

                target_x = 0.0

            elif locked_right:

                target_x = 1.0


            if locked_top:

                target_y = 0.0

            elif locked_bottom:

                target_y = 1.0


            # =================================================
            # CANVAS COORDINATES
            # =================================================

            target_canvas_x = (
                target_x
                *
                (CANVAS_WIDTH - 1)
            )

            target_canvas_y = (
                target_y
                *
                (CANVAS_HEIGHT - 1)
            )


            # =================================================
            # SMOOTH POSITION
            # =================================================

            if filtered_x is None:

                filtered_x = target_canvas_x

                filtered_y = target_canvas_y

            else:

                filtered_x += (
                    target_canvas_x
                    -
                    filtered_x
                ) * SMOOTHING

                filtered_y += (
                    target_canvas_y
                    -
                    filtered_y
                ) * SMOOTHING


            # =================================================
            # FINAL CURSOR
            # =================================================

            cursor_x = int(
                clamp(
                    filtered_x,
                    0,
                    CANVAS_WIDTH - 1
                )
            )

            cursor_y = int(
                clamp(
                    filtered_y,
                    0,
                    CANVAS_HEIGHT - 1
                )
            )


            current_point = (
                cursor_x,
                cursor_y
            )


            # =================================================
            # PINCH DETECTION
            # =================================================

            pinch_distance = math.sqrt(

                (thumb.x - index.x) ** 2

                +

                (thumb.y - index.y) ** 2
            )


            # =================================================
            # PALM SCALE
            # =================================================

            wrist = hand[0]

            index_mcp = hand[5]


            palm_scale = math.sqrt(

                (wrist.x - index_mcp.x) ** 2

                +

                (wrist.y - index_mcp.y) ** 2
            )


            if palm_scale > 0:

                touch_ratio = (
                    pinch_distance
                    /
                    palm_scale
                )

            else:

                touch_ratio = 1.0


            # =================================================
            # START DRAW
            # =================================================

            if not drawing:

                if touch_ratio < TOUCH_START:

                    drawing = True

                    previous_point = current_point


            # =================================================
            # CONTINUE DRAWING
            # =================================================

            else:

                if touch_ratio < TOUCH_STOP:

                    if previous_point is not None:

                        movement = distance(
                            previous_point,
                            current_point
                        )


                        # -------------------------------------
                        # NORMAL MOVEMENT
                        # -------------------------------------

                        if (
                            movement >= MIN_MOVEMENT
                            and
                            movement <= MAX_JUMP
                        ):

                            # Interpolate larger gaps
                            steps = max(
                                1,
                                int(
                                    movement / 4
                                )
                            )


                            for i in range(
                                1,
                                steps + 1
                            ):

                                t = (
                                    i
                                    /
                                    steps
                                )


                                px = int(
                                    previous_point[0]
                                    +
                                    (
                                        current_point[0]
                                        -
                                        previous_point[0]
                                    )
                                    * t
                                )


                                py = int(
                                    previous_point[1]
                                    +
                                    (
                                        current_point[1]
                                        -
                                        previous_point[1]
                                    )
                                    * t
                                )


                                cv2.circle(
                                    canvas,
                                    (
                                        px,
                                        py
                                    ),
                                    BRUSH_SIZE,
                                    (
                                        0,
                                        0,
                                        0
                                    ),
                                    -1,
                                    cv2.LINE_AA
                                )


                            previous_point = current_point


                        # -------------------------------------
                        # LARGE TRACKING JUMP
                        # -------------------------------------

                        elif movement > MAX_JUMP:

                            previous_point = current_point


                # =================================================
                # RELEASE
                # =================================================

                else:

                    drawing = False

                    previous_point = None


            # =================================================
            # CAMERA INDEX DOT
            # =================================================

            camera_x = int(
                raw_x * CAMERA_WIDTH
            )

            camera_y = int(
                raw_y * CAMERA_HEIGHT
            )


            cv2.circle(
                frame,
                (
                    camera_x,
                    camera_y
                ),
                8,
                (
                    0,
                    255,
                    0
                ),
                -1,
                cv2.LINE_AA
            )


            # =================================================
            # THUMB
            # =================================================

            thumb_x = int(
                thumb.x * CAMERA_WIDTH
            )

            thumb_y = int(
                thumb.y * CAMERA_HEIGHT
            )


            cv2.circle(
                frame,
                (
                    thumb_x,
                    thumb_y
                ),
                7,
                (
                    255,
                    0,
                    0
                ),
                -1,
                cv2.LINE_AA
            )


            # =================================================
            # THUMB-INDEX LINE
            # =================================================

            cv2.line(
                frame,
                (
                    thumb_x,
                    thumb_y
                ),
                (
                    camera_x,
                    camera_y
                ),
                (
                    255,
                    255,
                    0
                ),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # STATUS
            # =================================================

            status = (
                "DRAWING"
                if drawing
                else
                "MOVE"
            )


            cv2.putText(
                frame,
                status,
                (
                    20,
                    70
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # TOUCH VALUE
            # =================================================

            cv2.putText(
                frame,
                f"Touch: {touch_ratio:.2f}",
                (
                    20,
                    105
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # EDGE STATUS
            # =================================================

            edge_status = ""

            if locked_left:
                edge_status = "LEFT LOCK"

            elif locked_right:
                edge_status = "RIGHT LOCK"

            elif locked_top:
                edge_status = "TOP LOCK"

            elif locked_bottom:
                edge_status = "BOTTOM LOCK"


            if edge_status:

                cv2.putText(
                    frame,
                    edge_status,
                    (
                        20,
                        140
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (
                        0,
                        255,
                        255
                    ),
                    2,
                    cv2.LINE_AA
                )


            # =================================================
            # CANVAS CURSOR
            # =================================================

            cv2.circle(
                display_canvas,
                current_point,
                9,
                (
                    0,
                    0,
                    0
                ),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # HAND LOST
        # ====================================================

        else:

            lost_frames += 1


            if lost_frames > MAX_LOST_FRAMES:

                drawing = False

                previous_point = None

                x_history.clear()

                y_history.clear()


            cv2.putText(
                frame,
                "HAND SEARCHING...",
                (
                    20,
                    40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    0,
                    0,
                    255
                ),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # HEADERS
        # ====================================================

        cv2.putText(
            frame,
            "CAMERA",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            display_canvas,
            "AIRNOTE",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                0,
                0,
                0
            ),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "INDEX = MOVE",
            (
                20,
                CAMERA_HEIGHT - 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "THUMB + INDEX TOUCH = DRAW",
            (
                20,
                CAMERA_HEIGHT - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            display_canvas,
            "C = CLEAR    Q = QUIT",
            (
                CANVAS_WIDTH - 180,
                CANVAS_HEIGHT - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                0,
                0,
                0
            ),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SIDE BY SIDE
        # ====================================================

        output = np.hstack(
            (
                frame,
                display_canvas
            )
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AirNote - Smooth Draw V6",
            output
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break


        if key == ord("c"):

            canvas[:] = 255

            drawing = False

            previous_point = None


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("AirNote Smooth Draw V6 finished.")
