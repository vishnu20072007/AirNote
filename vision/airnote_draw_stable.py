import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AIRNOTE - FINAL STABLE DRAW + ERASER
#
# DRAW:
#   Thumb + Index pinch
#
# ERASER:
#   Index + Middle fingers raised and close
#   Ring + Pinky folded
#
# NORMAL HAND:
#   IDLE - NO DRAWING
# ============================================================


# ============================================================
# CAMERA
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# CANVAS
# ============================================================

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 700


# ============================================================
# DRAW SETTINGS
# ============================================================

BRUSH_SIZE = 3

MIN_MOVEMENT = 1.5
MAX_JUMP = 70


# ============================================================
# ERASER SETTINGS
# ============================================================

ERASER_SIZE = 55

ERASER_CONFIRM_FRAMES = 3
ERASER_RELEASE_FRAMES = 4


# ============================================================
# SMOOTHING
# ============================================================

SMOOTHING = 0.45
HISTORY_SIZE = 3


# ============================================================
# PINCH SETTINGS
# ============================================================

TOUCH_START = 0.28
TOUCH_STOP = 0.36


# ============================================================
# EDGE SETTINGS
# ============================================================

EDGE_START = 0.045
EDGE_RELEASE = 0.075


# ============================================================
# TRACKING
# ============================================================

MAX_LOST_FRAMES = 6


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

MODEL_PATH = "vision/models/hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.65
)


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


# ============================================================
# JOINT ANGLE
# ============================================================

def joint_angle(a, b, c):

    ba = np.array([
        a.x - b.x,
        a.y - b.y
    ])

    bc = np.array([
        c.x - b.x,
        c.y - b.y
    ])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0

    cosine = np.dot(
        ba,
        bc
    ) / (
        norm_ba * norm_bc
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# PINCH DETECTION
# ============================================================

def is_pinch(hand):

    thumb_tip = hand[4]
    index_tip = hand[8]

    palm_size = distance(
        hand[0],
        hand[9]
    )

    if palm_size < 0.001:
        return False

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    ratio = pinch_distance / palm_size

    return ratio <= TOUCH_START


# ============================================================
# PINCH RELEASE
# ============================================================

def pinch_released(hand):

    thumb_tip = hand[4]
    index_tip = hand[8]

    palm_size = distance(
        hand[0],
        hand[9]
    )

    if palm_size < 0.001:
        return True

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    ratio = pinch_distance / palm_size

    return ratio >= TOUCH_STOP


# ============================================================
# ERASER GESTURE
# ============================================================

def is_eraser_gesture(hand):

    """
    ERASER:

        INDEX + MIDDLE

             ☝️
             ☝️

    Index  = raised
    Middle = raised
    Ring   = folded
    Pinky  = folded

    The index and middle fingertips
    should be reasonably close.

    Normal hand movement:
        NOT ERASER

    Index only:
        NOT ERASER

    Pinch:
        DRAW
    """


    # --------------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------------

    wrist = hand[0]

    index_mcp = hand[5]
    index_pip = hand[6]
    index_tip = hand[8]

    middle_mcp = hand[9]
    middle_pip = hand[10]
    middle_tip = hand[12]

    ring_mcp = hand[13]
    ring_pip = hand[14]
    ring_tip = hand[16]

    pinky_mcp = hand[17]
    pinky_pip = hand[18]
    pinky_tip = hand[20]


    # --------------------------------------------------------
    # PALM SIZE
    # --------------------------------------------------------

    palm_size = distance(
        wrist,
        middle_mcp
    )

    if palm_size < 0.001:
        return False


    # ========================================================
    # INDEX
    # ========================================================

    index_angle = joint_angle(
        index_mcp,
        index_pip,
        index_tip
    )

    index_extended = (
        index_angle > 145
    )


    # Distance check
    index_length = distance(
        index_mcp,
        index_tip
    )

    index_reference = distance(
        index_mcp,
        index_pip
    )

    index_long = (
        index_length >
        index_reference * 1.25
    )


    # ========================================================
    # MIDDLE
    # ========================================================

    middle_angle = joint_angle(
        middle_mcp,
        middle_pip,
        middle_tip
    )

    middle_extended = (
        middle_angle > 145
    )


    middle_length = distance(
        middle_mcp,
        middle_tip
    )

    middle_reference = distance(
        middle_mcp,
        middle_pip
    )

    middle_long = (
        middle_length >
        middle_reference * 1.25
    )


    # ========================================================
    # RING FINGER FOLDED
    # ========================================================

    ring_angle = joint_angle(
        ring_mcp,
        ring_pip,
        ring_tip
    )

    ring_folded = (
        ring_angle < 150
    )


    # ========================================================
    # PINKY FOLDED
    # ========================================================

    pinky_angle = joint_angle(
        pinky_mcp,
        pinky_pip,
        pinky_tip
    )

    pinky_folded = (
        pinky_angle < 150
    )


    # ========================================================
    # INDEX + MIDDLE CLOSE
    # ========================================================

    index_middle_distance = distance(
        index_tip,
        middle_tip
    )

    fingers_close = (
        index_middle_distance <
        palm_size * 1.05
    )


    # ========================================================
    # FINAL ERASER CHECK
    # ========================================================

    return (
        index_extended
        and index_long
        and
        middle_extended
        and middle_long
        and
        ring_folded
        and
        pinky_folded
        and
        fingers_close
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    raise SystemExit


# ============================================================
# CANVAS
# ============================================================

canvas = np.ones(
    (
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        3
    ),
    dtype=np.uint8
) * 255


# ============================================================
# POSITION HISTORY
# ============================================================

history_x = deque(
    maxlen=HISTORY_SIZE
)

history_y = deque(
    maxlen=HISTORY_SIZE
)


# ============================================================
# TRACKING STATE
# ============================================================

smooth_x = None
smooth_y = None

previous_point = None

drawing = False
erasing = False

lost_frames = 0

timestamp_ms = 0


# ============================================================
# EDGE LOCK
# ============================================================

locked_left = False
locked_right = False
locked_top = False
locked_bottom = False


# ============================================================
# ERASER STATE
# ============================================================

eraser_candidate_frames = 0
eraser_release_counter = 0


# ============================================================
# MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ====================================================
        # CAMERA FRAME
        # ====================================================

        success, frame = cap.read()

        if not success:

            print("Camera frame failed.")

            break


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # IMPORTANT:
        # mp is imported at the top.
        # This prevents the previous NameError.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # DISPLAY CANVAS
        # ====================================================

        display_canvas = canvas.copy()

        gesture = "IDLE"

        cursor_x = None
        cursor_y = None


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            lost_frames = 0


            # =================================================
            # INDEX TIP
            # =================================================

            index_tip = hand[8]


            normalized_x = max(
                0.0,
                min(
                    1.0,
                    index_tip.x
                )
            )

            normalized_y = max(
                0.0,
                min(
                    1.0,
                    index_tip.y
                )
            )


            # =================================================
            # CAMERA -> CANVAS
            # =================================================

            target_x = int(
                normalized_x *
                (CANVAS_WIDTH - 1)
            )

            target_y = int(
                normalized_y *
                (CANVAS_HEIGHT - 1)
            )


            # =================================================
            # MEDIAN FILTER
            # =================================================

            history_x.append(
                target_x
            )

            history_y.append(
                target_y
            )

            median_x = int(
                np.median(
                    list(history_x)
                )
            )

            median_y = int(
                np.median(
                    list(history_y)
                )
            )


            # =================================================
            # SMOOTHING
            # =================================================

            if smooth_x is None:

                smooth_x = median_x
                smooth_y = median_y

            else:

                smooth_x = (
                    SMOOTHING *
                    median_x
                    +
                    (1.0 - SMOOTHING) *
                    smooth_x
                )

                smooth_y = (
                    SMOOTHING *
                    median_y
                    +
                    (1.0 - SMOOTHING) *
                    smooth_y
                )


            cursor_x = int(
                smooth_x
            )

            cursor_y = int(
                smooth_y
            )


            # =================================================
            # EDGE LOCKING
            # =================================================

            # LEFT
            if locked_left:

                cursor_x = 0

                if normalized_x > EDGE_RELEASE:

                    locked_left = False

            elif normalized_x < EDGE_START:

                cursor_x = 0

                locked_left = True


            # RIGHT
            if locked_right:

                cursor_x = CANVAS_WIDTH - 1

                if normalized_x < (
                    1.0 - EDGE_RELEASE
                ):

                    locked_right = False

            elif normalized_x > (
                1.0 - EDGE_START
            ):

                cursor_x = CANVAS_WIDTH - 1

                locked_right = True


            # TOP
            if locked_top:

                cursor_y = 0

                if normalized_y > EDGE_RELEASE:

                    locked_top = False

            elif normalized_y < EDGE_START:

                cursor_y = 0

                locked_top = True


            # BOTTOM
            if locked_bottom:

                cursor_y = CANVAS_HEIGHT - 1

                if normalized_y < (
                    1.0 - EDGE_RELEASE
                ):

                    locked_bottom = False

            elif normalized_y > (
                1.0 - EDGE_START
            ):

                cursor_y = CANVAS_HEIGHT - 1

                locked_bottom = True


            # =================================================
            # GESTURE DETECTION
            # =================================================

            pinch_now = is_pinch(
                hand
            )

            pinch_released_now = pinch_released(
                hand
            )

            eraser_now = is_eraser_gesture(
                hand
            )


            # =================================================
            # ERASER CANDIDATE
            # =================================================

            if eraser_now:

                eraser_candidate_frames += 1

                eraser_release_counter = 0

            else:

                eraser_candidate_frames = 0

                if erasing:

                    eraser_release_counter += 1

                else:

                    eraser_release_counter = 0


            # =================================================
            # ACTIVATE ERASER
            # =================================================

            if (
                not erasing
                and
                eraser_candidate_frames >=
                ERASER_CONFIRM_FRAMES
            ):

                erasing = True


            # =================================================
            # RELEASE ERASER
            # =================================================

            if (
                erasing
                and
                eraser_release_counter >=
                ERASER_RELEASE_FRAMES
            ):

                erasing = False

                eraser_release_counter = 0


            # =================================================
            # GESTURE PRIORITY
            #
            # 1. ERASER
            # 2. DRAW
            # 3. IDLE
            # =================================================

            if erasing:

                gesture = "ERASER"

                drawing = False

                previous_point = None


            elif pinch_now:

                gesture = "DRAW"

                drawing = True

                erasing = False


            else:

                gesture = "IDLE"

                drawing = False

                previous_point = None


            # =================================================
            # DRAW
            # =================================================

            if gesture == "DRAW":

                current_point = (
                    cursor_x,
                    cursor_y
                )


                if previous_point is not None:

                    dx = (
                        current_point[0]
                        -
                        previous_point[0]
                    )

                    dy = (
                        current_point[1]
                        -
                        previous_point[1]
                    )

                    movement = math.sqrt(
                        dx * dx +
                        dy * dy
                    )


                    # Prevent sudden jumps
                    if movement <= MAX_JUMP:

                        # Ignore tiny camera noise
                        if movement >= MIN_MOVEMENT:

                            cv2.line(
                                canvas,
                                previous_point,
                                current_point,
                                (0, 0, 0),
                                BRUSH_SIZE,
                                cv2.LINE_AA
                            )


                previous_point = current_point


            # =================================================
            # ERASER
            # =================================================

            elif gesture == "ERASER":

                previous_point = None


                # ------------------------------------------------
                # ERASER PREVIEW
                # ------------------------------------------------

                cv2.circle(
                    display_canvas,
                    (
                        cursor_x,
                        cursor_y
                    ),
                    ERASER_SIZE,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )


                # ------------------------------------------------
                # ACTUAL ERASE
                # ------------------------------------------------

                cv2.circle(
                    canvas,
                    (
                        cursor_x,
                        cursor_y
                    ),
                    ERASER_SIZE,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA
                )


            # =================================================
            # IDLE
            # =================================================

            else:

                previous_point = None


        # ====================================================
        # HAND LOST
        # ====================================================

        else:

            lost_frames += 1


            if lost_frames > MAX_LOST_FRAMES:

                drawing = False

                erasing = False

                previous_point = None

                history_x.clear()

                history_y.clear()

                smooth_x = None
                smooth_y = None

                eraser_candidate_frames = 0

                eraser_release_counter = 0


        # ====================================================
        # CAMERA PANEL
        # ====================================================

        camera_panel = np.zeros(
            (
                CANVAS_HEIGHT,
                500,
                3
            ),
            dtype=np.uint8
        )


        # ====================================================
        # CAMERA PREVIEW
        # ====================================================

        preview = cv2.resize(
            frame,
            (
                500,
                375
            )
        )

        preview_y = (
            CANVAS_HEIGHT - 375
        ) // 2


        camera_panel[
            preview_y:
            preview_y + 375,
            0:500
        ] = preview


        # ====================================================
        # CAMERA TITLE
        # ====================================================

        cv2.putText(
            camera_panel,
            "CAMERA",
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # AIRNOTE TITLE
        # ====================================================

        cv2.putText(
            display_canvas,
            "AIRNOTE",
            (35, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (70, 70, 70),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # STATUS
        # ====================================================

        if gesture == "DRAW":

            status_text = "DRAW"

            status_color = (
                0,
                150,
                0
            )

        elif gesture == "ERASER":

            status_text = "ERASER"

            status_color = (
                0,
                0,
                220
            )

        else:

            status_text = "IDLE"

            status_color = (
                100,
                100,
                100
            )


        cv2.putText(
            display_canvas,
            status_text,
            (820, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # COMBINE
        # ====================================================

        combined = np.hstack(
            (
                camera_panel,
                display_canvas
            )
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AIRNOTE",
            combined
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # C = CLEAR
        # ----------------------------------------------------

        if key == ord("c") or key == ord("C"):

            canvas[:] = 255

            previous_point = None


        # ----------------------------------------------------
        # Q = QUIT
        # ----------------------------------------------------

        if key == ord("q") or key == ord("Q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()