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
# CAMERA SETTINGS
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# CANVAS SETTINGS
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

ERASER_SIZE = 60

ERASER_CONFIRM_FRAMES = 2
ERASER_RELEASE_FRAMES = 3


# ============================================================
# SMOOTHING
# ============================================================

SMOOTHING = 0.55
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
# TWO FINGER ERASER
# ============================================================

def is_two_finger_eraser(hand):

    """
    ERASER GESTURE:

        INDEX + MIDDLE FINGERS UP

        ✌️

    Index  = UP
    Middle = UP
    Ring   = FOLDED
    Pinky  = FOLDED

    Thumb can be anywhere.
    """


    # ========================================================
    # INDEX
    # ========================================================

    index_mcp = hand[5]
    index_pip = hand[6]
    index_tip = hand[8]


    # ========================================================
    # MIDDLE
    # ========================================================

    middle_mcp = hand[9]
    middle_pip = hand[10]
    middle_tip = hand[12]


    # ========================================================
    # RING
    # ========================================================

    ring_mcp = hand[13]
    ring_pip = hand[14]
    ring_tip = hand[16]


    # ========================================================
    # PINKY
    # ========================================================

    pinky_mcp = hand[17]
    pinky_pip = hand[18]
    pinky_tip = hand[20]


    # ========================================================
    # INDEX
    # ========================================================

    index_angle = joint_angle(
        index_mcp,
        index_pip,
        index_tip
    )

    index_extended = (
        index_angle > 135
    )

    index_up = (
        index_tip.y <
        index_mcp.y
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
        middle_angle > 135
    )

    middle_up = (
        middle_tip.y <
        middle_mcp.y
    )


    # ========================================================
    # RING FOLDED
    # ========================================================

    ring_angle = joint_angle(
        ring_mcp,
        ring_pip,
        ring_tip
    )

    ring_folded = (
        ring_angle < 155
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
        pinky_angle < 155
    )


    # ========================================================
    # INDEX + MIDDLE SHOULD NOT BE TOO FAR
    # ========================================================

    palm_size = distance(
        hand[0],
        hand[9]
    )

    if palm_size < 0.001:
        return False

    index_middle_distance = distance(
        index_tip,
        middle_tip
    )

    fingers_close = (
        index_middle_distance <
        palm_size * 1.35
    )


    # ========================================================
    # FINAL ERASER GESTURE
    # ========================================================

    return (
        index_extended
        and
        middle_extended
        and
        index_up
        and
        middle_up
        and
        ring_folded
        and
        pinky_folded
        and
        fingers_close
    )


# ============================================================
# ERASE LINE BETWEEN TWO POINTS
# ============================================================

def erase_between_points(
    canvas,
    p1,
    p2,
    radius
):

    if p1 is None or p2 is None:

        cv2.circle(
            canvas,
            p2,
            radius,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        return


    x1, y1 = p1
    x2, y2 = p2


    distance_px = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


    # More points = no gaps during fast movement

    steps = max(
        1,
        int(distance_px / max(5, radius * 0.35))
    )


    for i in range(steps + 1):

        t = i / steps

        x = int(
            x1 +
            (x2 - x1) * t
        )

        y = int(
            y1 +
            (y2 - y1) * t
        )

        cv2.circle(
            canvas,
            (x, y),
            radius,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
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

previous_eraser_point = None

drawing = False

erasing = False

lost_frames = 0

timestamp_ms = 0


# ============================================================
# EDGE LOCK STATE
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
        # READ CAMERA
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


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # MEDIAPIPE DETECTION
        # ====================================================

        timestamp_ms += 33


        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # INITIAL DISPLAY CANVAS
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

                cursor_x = (
                    CANVAS_WIDTH - 1
                )

                if normalized_x < (
                    1.0 - EDGE_RELEASE
                ):

                    locked_right = False

            elif normalized_x > (
                1.0 - EDGE_START
            ):

                cursor_x = (
                    CANVAS_WIDTH - 1
                )

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

                cursor_y = (
                    CANVAS_HEIGHT - 1
                )

                if normalized_y < (
                    1.0 - EDGE_RELEASE
                ):

                    locked_bottom = False

            elif normalized_y > (
                1.0 - EDGE_START
            ):

                cursor_y = (
                    CANVAS_HEIGHT - 1
                )

                locked_bottom = True


            # =================================================
            # CAMERA INDEX DOT
            # =================================================

            camera_x = int(
                index_tip.x *
                CAMERA_WIDTH
            )


            camera_y = int(
                index_tip.y *
                CAMERA_HEIGHT
            )


            camera_x = max(
                0,
                min(
                    CAMERA_WIDTH - 1,
                    camera_x
                )
            )


            camera_y = max(
                0,
                min(
                    CAMERA_HEIGHT - 1,
                    camera_y
                )
            )


            cv2.circle(
                frame,
                (
                    camera_x,
                    camera_y
                ),
                7,
                (0, 255, 0),
                -1
            )


            # =================================================
            # GESTURE DETECTION
            # =================================================

            pinch_now = is_pinch(
                hand
            )


            pinch_released_now = pinch_released(
                hand
            )


            two_finger_eraser = is_two_finger_eraser(
                hand
            )


            # =================================================
            # DRAW PRIORITY
            # =================================================

            if pinch_now or (
                drawing
                and
                not pinch_released_now
            ):

                gesture = "DRAW"

                drawing = True

                erasing = False

                eraser_candidate_frames = 0

                eraser_release_counter = 0

                previous_eraser_point = None


            else:

                drawing = False

                previous_point = None


                # =================================================
                # ERASER DETECTION
                # =================================================

                if two_finger_eraser:

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
                # CURRENT MODE
                # =================================================

                if erasing:

                    gesture = "ERASER"

                else:

                    gesture = "IDLE"


            # ====================================================
            # DRAW MODE
            # ====================================================

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


            # ====================================================
            # ERASER MODE
            # ====================================================

            elif gesture == "ERASER":

                previous_point = None


                current_eraser_point = (
                    cursor_x,
                    cursor_y
                )


                # ------------------------------------------------
                # ACTUAL ERASE
                # ------------------------------------------------

                erase_between_points(
                    canvas,
                    previous_eraser_point,
                    current_eraser_point,
                    ERASER_SIZE
                )


                # ------------------------------------------------
                # SAVE CURRENT ERASER POSITION
                # ------------------------------------------------

                previous_eraser_point = (
                    current_eraser_point
                )


                # ------------------------------------------------
                # IMPORTANT:
                # REFRESH DISPLAY AFTER ERASE
                # ------------------------------------------------

                display_canvas = canvas.copy()


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


            # ====================================================
            # IDLE
            # ====================================================

            else:

                previous_point = None

                previous_eraser_point = None


                if cursor_x is not None:

                    cv2.circle(
                        display_canvas,
                        (
                            cursor_x,
                            cursor_y
                        ),
                        7,
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


        # ====================================================
        # STATUS TEXT
        # ====================================================

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
        # COMBINE CAMERA + CANVAS
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


        # ====================================================
        # C = CLEAR
        # ====================================================

        if key == ord("c") or key == ord("C"):

            canvas[:] = 255

            previous_point = None

            previous_eraser_point = None


        # ====================================================
        # Q = QUIT
        # ====================================================

        if key == ord("q") or key == ord("Q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()