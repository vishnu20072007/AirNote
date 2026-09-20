import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# AIRNOTE - CLEAN CURSOR TEST
# Camera -> Index Finger -> AirNote Canvas
# ============================================================

MODEL_PATH = "vision/models/hand_landmarker.task"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480

# Smoothing
SMOOTHING = 0.35

smooth_x = None
smooth_y = None


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
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
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
# HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Camera frame not received.")
            break

        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(
            frame,
            (CAMERA_WIDTH, CAMERA_HEIGHT)
        )


        # ====================================================
        # CREATE CLEAN WHITE CANVAS
        # ====================================================

        canvas = np.ones(
            (
                CANVAS_HEIGHT,
                CANVAS_WIDTH,
                3
            ),
            dtype=np.uint8
        ) * 255


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
            / cv2.getTickFrequency()
        )

        result = landmarker.detect_for_video(
            image,
            timestamp
        )


        # ====================================================
        # INDEX FINGERTIP
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Landmark 8 = Index fingertip
            index = hand[8]

            # ------------------------------------------------
            # NORMALIZED POSITION
            # ------------------------------------------------

            x = max(
                0.0,
                min(1.0, index.x)
            )

            y = max(
                0.0,
                min(1.0, index.y)
            )


            # ------------------------------------------------
            # CAMERA POSITION
            # ------------------------------------------------

            camera_x = int(
                x * (CAMERA_WIDTH - 1)
            )

            camera_y = int(
                y * (CAMERA_HEIGHT - 1)
            )


            # ------------------------------------------------
            # CANVAS POSITION
            # ------------------------------------------------

            target_x = int(
                x * (CANVAS_WIDTH - 1)
            )

            target_y = int(
                y * (CANVAS_HEIGHT - 1)
            )


            # ------------------------------------------------
            # SMOOTH CURSOR
            # ------------------------------------------------

            if smooth_x is None:

                smooth_x = target_x
                smooth_y = target_y

            else:

                smooth_x = (
                    smooth_x * (1 - SMOOTHING)
                    + target_x * SMOOTHING
                )

                smooth_y = (
                    smooth_y * (1 - SMOOTHING)
                    + target_y * SMOOTHING
                )


            cursor_x = int(smooth_x)
            cursor_y = int(smooth_y)


            # =================================================
            # CAMERA - GREEN INDEX DOT
            # =================================================

            cv2.circle(
                frame,
                (camera_x, camera_y),
                9,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                "INDEX",
                (
                    camera_x + 12,
                    camera_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # CANVAS - CURSOR
            # =================================================

            # Outer ring
            cv2.circle(
                canvas,
                (cursor_x, cursor_y),
                14,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Center dot
            cv2.circle(
                canvas,
                (cursor_x, cursor_y),
                5,
                (0, 0, 0),
                -1,
                cv2.LINE_AA
            )


            # =================================================
            # CROSSHAIR
            # =================================================

            cv2.line(
                canvas,
                (
                    cursor_x - 22,
                    cursor_y
                ),
                (
                    cursor_x + 22,
                    cursor_y
                ),
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

            cv2.line(
                canvas,
                (
                    cursor_x,
                    cursor_y - 22
                ),
                (
                    cursor_x,
                    cursor_y + 22
                ),
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )


            # =================================================
            # POSITION
            # =================================================

            cv2.putText(
                canvas,
                f"X: {cursor_x}  Y: {cursor_y}",
                (20, CANVAS_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )


        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # HEADERS
        # ====================================================

        cv2.putText(
            frame,
            "CAMERA",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            "AIRNOTE CANVAS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SIDE-BY-SIDE DISPLAY
        # ====================================================

        output = np.hstack(
            (
                frame,
                canvas
            )
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            output,
            "Move INDEX finger",
            (20, CAMERA_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            "Q = Quit",
            (
                CAMERA_WIDTH + 20,
                CAMERA_HEIGHT - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AirNote - Cursor Test",
            output
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("AirNote cursor test finished.")