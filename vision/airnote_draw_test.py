import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AIRNOTE
# PINCH = DRAW
# INDEX + MIDDLE = ERASER
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

    min_hand_detection_confidence=0.60,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.60
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

    return ratio < TOUCH_START


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

    return ratio > TOUCH_STOP


# ============================================================
# FINGER EXTENSION
# ============================================================

def finger_is_extended(
    hand,
    mcp_id,
    pip_id,
    tip_id
):

    mcp = hand[mcp_id]
    pip = hand[pip_id]
    tip = hand[tip_id]

    # Finger should be reasonably straight
    angle = joint_angle(
        mcp,
        pip,
        tip
    )

    straight = angle > 125

    # Finger tip should be above its MCP
    vertical = tip.y < mcp.y

    # Tip should be farther from wrist than PIP
    wrist = hand[0]

    tip_distance = distance(
        wrist,
        tip
    )

    pip_distance = distance(
        wrist,
        pip
    )

    farther = tip_distance > pip_distance * 0.95

    return (
        straight
        and
        vertical
        and
        farther
    )


# ============================================================
# FINGER FOLDED
# ============================================================

def finger_is_folded(
    hand,
    mcp_id,
    pip_id,
    tip_id
):

    mcp = hand[mcp_id]
    pip = hand[pip_id]
    tip = hand[tip_id]

    angle = joint_angle(
        mcp,
        pip,
        tip
    )

    # Either bent OR tip is not above MCP
    bent = angle < 160

    not_up = tip.y >= mcp.y

    return bent or not_up


# ============================================================
# ERASER GESTURE
# ============================================================

def is_eraser_gesture(hand):

    """
    ERASER:

        INDEX  = UP
        MIDDLE = UP
        RING   = FOLDED
        PINKY  = FOLDED

    Example:

             ☝️
             ☝️

        index + middle raised

    Pinch is NOT eraser.
    """

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index_up = finger_is_extended(
        hand,
        5,
        6,
        8
    )

    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    middle_up = finger_is_extended(
        hand,
        9,
        10,
        12
    )

    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring_folded = finger_is_folded(
        hand,
        13,
        14,
        16
    )

    # --------------------------------------------------------
    # PINKY
    # --------------------------------------------------------

    pinky_folded = finger_is_folded(
        hand,
        17,
        18,
        20
    )

    # --------------------------------------------------------
    # PINCH CHECK
    # --------------------------------------------------------

    # If thumb and index are very close,
    # give priority to DRAW.
    thumb_index_distance = distance(
        hand[4],
        hand[8]
    )

    palm_size = distance(
        hand[0],
        hand[9]
    )

    if palm_size < 0.001:
        return False

    pinch_ratio = (
        thumb_index_distance /
        palm_size
    )

    if pinch_ratio < TOUCH_START:
        return False

    # --------------------------------------------------------
    # INDEX + MIDDLE SHOULD BE SEPARATE ENOUGH
    # --------------------------------------------------------

    index_middle_distance = distance(
        hand[8],
        hand[12]
    )

    # Too close = probably pinch / unstable
    fingers_not_collapsed = (
        index_middle_distance >
        palm_size * 0.15
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    return (
        index_up
        and
        middle_up
        and
        ring_folded
        and
        pinky_folded
        and
        fingers_not_collapsed
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
# ERASER PATH
# ============================================================

previous_eraser_point = None


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
        # DISPLAY
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
                    SMOOTHING * median_x
                    +
                    (1.0 - SMOOTHING) * smooth_x
                )

                smooth_y = (
                    SMOOTHING * median_y
                    +
                    (1.0 - SMOOTHING) * smooth_y
                )


            cursor_x = int(
                smooth_x
            )

            cursor_y = int(
                smooth_y
            )


            # =================================================
            # EDGE LOCK - LEFT
            # =================================================

            if locked_left:

                cursor_x = 0

                if normalized_x > EDGE_RELEASE:

                    locked_left = False

            elif normalized_x < EDGE_START:

                cursor_x = 0

                locked_left = True


            # =================================================
            # EDGE LOCK - RIGHT
            # =================================================

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


            # =================================================
            # EDGE LOCK - TOP
            # =================================================

            if locked_top:

                cursor_y = 0

                if normalized_y > EDGE_RELEASE:

                    locked_top = False

            elif normalized_y < EDGE_START:

                cursor_y = 0

                locked_top = True


            # =================================================
            # EDGE LOCK - BOTTOM
            # =================================================

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
            # DRAW
            # =================================================

            if pinch_now:

                gesture = "DRAW"

                drawing = True

                erasing = False

                previous_eraser_point = None

                eraser_candidate_frames = 0
                eraser_release_counter = 0


            # =================================================
            # CONTINUE DRAW ONLY WHILE PINCH IS STILL HELD
            # =================================================

            elif (
                drawing
                and
                not pinch_released_now
            ):

                gesture = "DRAW"

                drawing = True

                erasing = False

                previous_eraser_point = None


            # =================================================
            # NOT DRAWING
            # =================================================

            else:

                drawing = False

                previous_point = None


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
                    eraser_candidate_frames
                    >= ERASER_CONFIRM_FRAMES
                ):

                    erasing = True

                    previous_eraser_point = None


                # =================================================
                # RELEASE ERASER
                # =================================================

                if (
                    erasing
                    and
                    eraser_release_counter
                    >= ERASER_RELEASE_FRAMES
                ):

                    erasing = False

                    eraser_release_counter = 0

                    previous_eraser_point = None


                # =================================================
                # MODE
                # =================================================

                if erasing:

                    gesture = "ERASER"

                else:

                    gesture = "IDLE"


            # =================================================
            # DRAW MODE
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


                    # Ignore huge tracking jumps
                    if movement <= MAX_JUMP:

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

                previous_eraser_point = None


            # =================================================
            # ERASER MODE
            # =================================================

            elif gesture == "ERASER":

                previous_point = None


                current_eraser_point = (
                    cursor_x,
                    cursor_y
                )


                # =================================================
                # ERASER PREVIEW
                # =================================================

                cv2.circle(
                    display_canvas,
                    current_eraser_point,
                    ERASER_SIZE,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )


                # =================================================
                # ACTUAL ERASING
                #
                # IMPORTANT:
                # Use a thick LINE between old and new point.
                # This prevents gaps when hand moves quickly.
                # =================================================

                if previous_eraser_point is not None:

                    cv2.line(
                        canvas,
                        previous_eraser_point,
                        current_eraser_point,
                        (255, 255, 255),
                        ERASER_SIZE * 2,
                        cv2.LINE_AA
                    )


                # =================================================
                # ERASE CURRENT POSITION
                # =================================================

                cv2.circle(
                    canvas,
                    current_eraser_point,
                    ERASER_SIZE,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA
                )


                previous_eraser_point = (
                    current_eraser_point
                )


            # =================================================
            # IDLE
            # =================================================

            else:

                previous_point = None

                previous_eraser_point = None


                # Small cursor preview
                if cursor_x is not None:

                    cv2.circle(
                        display_canvas,
                        (
                            cursor_x,
                            cursor_y
                        ),
                        6,
                        (120, 120, 120),
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

                erasing = False

                previous_point = None

                previous_eraser_point = None

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

            previous_eraser_point = None


        # ----------------------------------------------------
        # Q = QUIT
        # ----------------------------------------------------

        if key == ord("q") or key == ord("Q"):

            break


# ============================================================
# CLEANUP
# ===================== =======================================

cap.release()

cv2.destroyAllWindows()
