import cv2
import mediapipe as mp

# ============================================================
# AIRNOTE - CAMERA TO CANVAS MAPPING TEST
# ============================================================

MODEL_PATH = "vision/models/hand_landmarker.task"

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit


# ------------------------------------------------------------
# MEDIAPIPE
# ------------------------------------------------------------

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
# CANVAS
# ============================================================

CAMERA_W = 640
CAMERA_H = 480

CANVAS_W = 640
CANVAS_H = 480

WINDOW_W = CAMERA_W + CANVAS_W
WINDOW_H = CAMERA_H


# ------------------------------------------------------------
# SMOOTHING
# ------------------------------------------------------------

smooth_x = None
smooth_y = None

SMOOTHING = 0.45


# ============================================================
# START
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            break


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)


        # Resize camera preview

        camera_view = cv2.resize(
            frame,
            (CAMERA_W, CAMERA_H)
        )


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = int(
            cv2.getTickCount()
            * 1000
            / cv2.getTickFrequency()
        )


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            image,
            timestamp
        )


        # ====================================================
        # CREATE CLEAN AIRNOTE CANVAS
        # ====================================================

        canvas = 255 * (
            cv2.UMat(
                CANVAS_H,
                CANVAS_W,
                cv2.CV_8UC3
            )
        ).get()


        # ----------------------------------------------------
        # CANVAS TITLE
        # ----------------------------------------------------

        cv2.putText(
            canvas,
            "AIRNOTE CANVAS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Index fingertip = landmark 8
            index = hand[8]


            # ------------------------------------------------
            # CAMERA COORDINATES
            # ------------------------------------------------

            raw_x = index.x
            raw_y = index.y


            # Keep normalized values inside screen

            raw_x = max(
                0.0,
                min(1.0, raw_x)
            )

            raw_y = max(
                0.0,
                min(1.0, raw_y)
            )


            # ------------------------------------------------
            # MAP CAMERA -> CANVAS
            # ------------------------------------------------

            target_x = int(
                raw_x * (CANVAS_W - 1)
            )

            target_y = int(
                raw_y * (CANVAS_H - 1)
            )


            # ------------------------------------------------
            # SMOOTH POSITION
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


            mapped_x = int(smooth_x)
            mapped_y = int(smooth_y)


            # =================================================
            # SHOW INDEX ON CAMERA
            # =================================================

            camera_x = int(
                index.x * frame.shape[1]
            )

            camera_y = int(
                index.y * frame.shape[0]
            )


            cv2.circle(
                camera_view,
                (
                    int(camera_x * CAMERA_W / frame.shape[1]),
                    int(camera_y * CAMERA_H / frame.shape[0])
                ),
                10,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )


            cv2.putText(
                camera_view,
                "INDEX",
                (
                    int(camera_x * CAMERA_W / frame.shape[1]) + 12,
                    int(camera_y * CAMERA_H / frame.shape[0])
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # SHOW MAPPED DOT ON CANVAS
            # =================================================

            cv2.circle(
                canvas,
                (mapped_x, mapped_y),
                12,
                (0, 0, 255),
                -1,
                cv2.LINE_AA
            )


            cv2.circle(
                canvas,
                (mapped_x, mapped_y),
                22,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                canvas,
                "MAPPED",
                (
                    mapped_x + 18,
                    mapped_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


            # ------------------------------------------------
            # COORDINATES
            # ------------------------------------------------

            cv2.putText(
                canvas,
                f"X: {mapped_x}  Y: {mapped_y}",
                (20, CANVAS_H - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )


        else:

            cv2.putText(
                camera_view,
                "NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # DIVIDER
        # ====================================================

        divider = 255 * (
            cv2.UMat(
                WINDOW_H,
                4,
                cv2.CV_8UC3
            )
        ).get()


        # ====================================================
        # COMBINE CAMERA + CANVAS
        # ====================================================

        output = cv2.hconcat(
            [
                camera_view,
                canvas
            ]
        )


        # ====================================================
        # LABELS
        # ====================================================

        cv2.putText(
            output,
            "CAMERA",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            output,
            "MOVE INDEX FINGER",
            (
                CAMERA_W + 20,
                65
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            output,
            "Q = QUIT",
            (
                WINDOW_W - 120,
                WINDOW_H - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AirNote - Mapping Test",
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

print("Mapping test finished.")