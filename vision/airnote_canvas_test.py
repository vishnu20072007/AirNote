import cv2
import mediapipe as mp
import time
import math
import json
import os
import numpy as np


# ============================================================
# AIRNOTE - PRECISION V7
#
# 🤏 Thumb + Index TOUCH  -> DRAW
# ☝️ Index Only            -> ERASER
# ✊ Fist                  -> NOTHING
# ✋ Open Hand             -> NOTHING
#
# Camera + hand landmarks are visible.
# Calibration is loaded automatically.
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "vision/models/hand_landmarker.task"
CALIBRATION_PATH = "vision/calibration.json"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Hand model not found:")
    print(MODEL_PATH)
    raise SystemExit


if not os.path.exists(CALIBRATION_PATH):
    print("ERROR: Calibration file not found.")
    print()
    print("Run:")
    print("python vision/calibration_test.py")
    raise SystemExit


# ============================================================
# LOAD CALIBRATION
# ============================================================

with open(CALIBRATION_PATH, "r") as f:
    calibration = json.load(f)


points = calibration["points"]


src_points = np.float32([
    points["top_left"],
    points["top_right"],
    points["bottom_right"],
    points["bottom_left"]
])


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit


camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


ret, frame = camera.read()

if not ret:
    print("ERROR: Could not read webcam.")
    camera.release()
    raise SystemExit


frame_height, frame_width = frame.shape[:2]


# ============================================================
# DRAWING CANVAS
# ============================================================

canvas = np.ones(
    (
        frame_height,
        frame_width,
        3
    ),
    dtype=np.uint8
) * 255


# ============================================================
# PERSPECTIVE MAPPING
# ============================================================

dst_points = np.float32([
    [0, 0],
    [frame_width - 1, 0],
    [frame_width - 1, frame_height - 1],
    [0, frame_height - 1]
])


perspective_matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)


# ============================================================
# DRAW SETTINGS
# ============================================================

BRUSH_SIZE = 4
ERASER_SIZE = 55


# ============================================================
# TOUCH SETTINGS
# ============================================================

# Thumb + index must be VERY close.

TOUCH_START = 0.27
TOUCH_STOP = 0.34

TOUCH_CONFIRM_FRAMES = 2

touch_frames = 0


# ============================================================
# ERASER SETTINGS
# ============================================================

ERASER_CONFIRM_FRAMES = 3
ERASER_RELEASE_FRAMES = 4

eraser_frames = 0
eraser_release_counter = 0


# ============================================================
# MODE
# ============================================================

mode = "IDLE"


# ============================================================
# LOW LATENCY SMOOTHER
# ============================================================

class SmoothPoint:

    def __init__(self):

        self.x = None
        self.y = None


    def update(self, x, y):

        if self.x is None:

            self.x = float(x)
            self.y = float(y)

            return self.x, self.y


        movement = math.hypot(
            x - self.x,
            y - self.y
        )


        # Very small movement:
        # strong stabilization

        if movement < 3:

            response = 0.35


        # Normal movement:
        # balanced

        elif movement < 12:

            response = 0.70


        # Fast movement:
        # low latency

        else:

            response = 0.92


        self.x += (
            x - self.x
        ) * response


        self.y += (
            y - self.y
        ) * response


        return self.x, self.y


    def reset(self):

        self.x = None
        self.y = None


cursor = SmoothPoint()


# ============================================================
# STROKE STATE
# ============================================================

previous_x = None
previous_y = None


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


# ============================================================
# TOUCH RATIO
# ============================================================

def touch_ratio(hand):

    thumb = hand[4]
    index = hand[8]

    wrist = hand[0]
    middle_mcp = hand[9]


    finger_distance = distance(
        thumb,
        index
    )


    palm_size = distance(
        wrist,
        middle_mcp
    )


    if palm_size <= 0:
        return 999


    return (
        finger_distance
        / palm_size
    )


# ============================================================
# FINGER EXTENSION
# ============================================================

def is_extended(
    hand,
    tip_id,
    pip_id,
    mcp_id
):

    wrist = hand[0]


    tip_distance = distance(
        hand[tip_id],
        wrist
    )


    pip_distance = distance(
        hand[pip_id],
        wrist
    )


    mcp_distance = distance(
        hand[mcp_id],
        wrist
    )


    return (
        tip_distance
        > pip_distance
        > mcp_distance
    )


# ============================================================
# INDEX ONLY
# ============================================================

def index_only(hand):

    index = is_extended(
        hand,
        8,
        6,
        5
    )


    middle = is_extended(
        hand,
        12,
        10,
        9
    )


    ring = is_extended(
        hand,
        16,
        14,
        13
    )


    pinky = is_extended(
        hand,
        20,
        18,
        17
    )


    return (
        index
        and not middle
        and not ring
        and not pinky
    )


# ============================================================
# MAP POINT
# ============================================================

def map_point(x, y):

    point = np.float32([
        [
            [x, y]
        ]
    ])


    mapped = cv2.perspectiveTransform(
        point,
        perspective_matrix
    )


    new_x = float(
        mapped[0, 0, 0]
    )


    new_y = float(
        mapped[0, 0, 1]
    )


    new_x = max(
        0,
        min(
            frame_width - 1,
            new_x
        )
    )


    new_y = max(
        0,
        min(
            frame_height - 1,
            new_y
        )
    )


    return new_x, new_y


# ============================================================
# RESET DRAWING CURSOR
# ============================================================

def reset_cursor():

    global previous_x
    global previous_y

    previous_x = None
    previous_y = None

    cursor.reset()


# ============================================================
# DRAW LINE
# ============================================================

def draw_line(x, y):

    global previous_x
    global previous_y


    if previous_x is None:

        previous_x = x
        previous_y = y

        return


    movement = math.hypot(
        x - previous_x,
        y - previous_y
    )


    # Ignore tiny noise

    if movement < 0.8:
        return


    # Number of interpolation points

    steps = max(
        1,
        int(
            movement / 2.5
        )
    )


    last_x = previous_x
    last_y = previous_y


    for i in range(
        1,
        steps + 1
    ):

        t = i / steps


        new_x = (
            last_x
            + (x - last_x) * t
        )


        new_y = (
            last_y
            + (y - last_y) * t
        )


        cv2.line(
            canvas,
            (
                int(last_x),
                int(last_y)
            ),
            (
                int(new_x),
                int(new_y)
            ),
            (0, 0, 0),
            BRUSH_SIZE,
            cv2.LINE_AA
        )


        last_x = new_x
        last_y = new_y


    previous_x = x
    previous_y = y


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
# FPS
# ============================================================

fps = 0
fps_count = 0
fps_start = time.time()


# ============================================================
# MAIN
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        success, frame = camera.read()


        if not success:
            break


        frame = cv2.flip(
            frame,
            1
        )


        # Keep a copy for camera display

        camera_view = frame.copy()


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        timestamp_ms = int(
            time.monotonic() * 1000
        )


        results = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        detected_mode = "IDLE"


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]


            # ------------------------------------------------
            # Gesture measurements
            # ------------------------------------------------

            ratio = touch_ratio(hand)

            eraser_gesture = index_only(hand)


            # ------------------------------------------------
            # Touch confirmation
            # ------------------------------------------------

            if ratio < TOUCH_START:

                touch_frames += 1

            elif ratio > TOUCH_STOP:

                touch_frames = 0


            # ------------------------------------------------
            # Eraser confirmation
            # ------------------------------------------------

            if eraser_gesture:

                eraser_frames += 1
                eraser_release_counter = 0

            else:

                eraser_frames = 0

                if mode == "ERASER":

                    eraser_release_counter += 1


            # =================================================
            # PRIORITY
            # =================================================

            if (
                touch_frames
                >= TOUCH_CONFIRM_FRAMES
            ):

                detected_mode = "DRAW"


            elif (
                eraser_frames
                >= ERASER_CONFIRM_FRAMES
            ):

                detected_mode = "ERASER"


            elif mode == "ERASER":

                if (
                    eraser_release_counter
                    < ERASER_RELEASE_FRAMES
                ):

                    detected_mode = "ERASER"

                else:

                    detected_mode = "IDLE"


            else:

                detected_mode = "IDLE"


            # =================================================
            # MODE CHANGE
            # =================================================

            if detected_mode != mode:

                mode = detected_mode

                reset_cursor()


            # =================================================
            # INDEX TIP
            # =================================================

            index_tip = hand[8]


            camera_x = (
                index_tip.x
                * frame_width
            )


            camera_y = (
                index_tip.y
                * frame_height
            )


            # =================================================
            # MAP USING CALIBRATION
            # =================================================

            mapped_x, mapped_y = map_point(
                camera_x,
                camera_y
            )


            # =================================================
            # SMOOTH
            # =================================================

            smooth_x, smooth_y = cursor.update(
                mapped_x,
                mapped_y
            )


            current_x = int(
                max(
                    0,
                    min(
                        frame_width - 1,
                        smooth_x
                    )
                )
            )


            current_y = int(
                max(
                    0,
                    min(
                        frame_height - 1,
                        smooth_y
                    )
                )
            )


            # =================================================
            # DRAW
            # =================================================

            if mode == "DRAW":

                draw_line(
                    current_x,
                    current_y
                )


            # =================================================
            # ERASER
            # =================================================

            elif mode == "ERASER":

                cv2.circle(
                    canvas,
                    (
                        current_x,
                        current_y
                    ),
                    ERASER_SIZE,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA
                )


            # =================================================
            # DRAWING CURSOR
            # =================================================

            if mode == "DRAW":

                cv2.circle(
                    camera_view,
                    (
                        int(camera_x),
                        int(camera_y)
                    ),
                    8,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )


            elif mode == "ERASER":

                cv2.circle(
                    camera_view,
                    (
                        int(camera_x),
                        int(camera_y)
                    ),
                    35,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )


            # =================================================
            # LANDMARKS
            # =================================================

            for landmark in hand:

                lx = int(
                    landmark.x
                    * frame_width
                )

                ly = int(
                    landmark.y
                    * frame_height
                )


                cv2.circle(
                    camera_view,
                    (
                        lx,
                        ly
                    ),
                    3,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )


            # =================================================
            # DEBUG TOUCH VALUE
            # =================================================

            cv2.putText(
                camera_view,
                f"Touch: {ratio:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            mode = "IDLE"

            touch_frames = 0
            eraser_frames = 0
            eraser_release_counter = 0

            reset_cursor()


        # ====================================================
        # DRAW CALIBRATION BORDER ON CAMERA VIEW
        # ====================================================

        calibration_display = src_points.astype(
            np.int32
        )


        cv2.polylines(
            camera_view,
            [calibration_display],
            True,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.rectangle(
            camera_view,
            (0, 0),
            (300, 75),
            (30, 30, 30),
            -1
        )


        cv2.putText(
            camera_view,
            f"MODE: {mode}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            camera_view,
            "Camera / Hand Tracking",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # FPS
        # ====================================================

        fps_count += 1

        elapsed = (
            time.time()
            - fps_start
        )


        if elapsed >= 1:

            fps = (
                fps_count
                / elapsed
            )

            fps_count = 0
            fps_start = time.time()


        cv2.putText(
            camera_view,
            f"FPS: {fps:.0f}",
            (
                frame_width - 120,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # COMBINE CAMERA + CANVAS SIDE BY SIDE
        # ====================================================

        left = cv2.resize(
            camera_view,
            (
                frame_width // 2,
                frame_height // 2
            )
        )


        right = cv2.resize(
            canvas,
            (
                frame_width // 2,
                frame_height // 2
            )
        )


        # Labels

        cv2.putText(
            left,
            "CAMERA",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            right,
            "AIRNOTE CANVAS",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )


        output = np.hstack(
            (
                left,
                right
            )
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            output,
            "Touch Thumb + Index = DRAW",
            (
                15,
                output.shape[0] - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            output,
            "Index Only = ERASER | C = Clear | Q = Quit",
            (
                15,
                output.shape[0] - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "AirNote - Precision V7",
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

            reset_cursor()


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("AirNote Precision V7 finished.")