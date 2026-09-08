import cv2
import mediapipe as mp
import time
import math

# -----------------------------
# MediaPipe setup
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="vision/models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -----------------------------
# Webcam
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open webcam")
    exit()

# -----------------------------
# Drawing canvas
# -----------------------------

canvas = None

previous_point = None
smooth_x = None
smooth_y = None

# Smoothing strength
SMOOTHING = 0.35

# Ignore tiny movements
MIN_MOVEMENT = 3

start_time = time.time()

# -----------------------------
# Hand tracking
# -----------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Could not read webcam frame")
            break

        frame = cv2.flip(frame, 1)

        # Create white canvas
        if canvas is None:
            canvas = frame.copy()
            canvas[:] = 255

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        drawing = False

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            # Index fingertip
            index_tip = hand[8]

            # Index joint
            index_joint = hand[6]

            # Raw fingertip position
            raw_x = int(index_tip.x * frame.shape[1])
            raw_y = int(index_tip.y * frame.shape[0])

            # Check if index finger is raised
            if index_tip.y < index_joint.y:

                drawing = True

                # -----------------------------
                # Smooth fingertip movement
                # -----------------------------

                if smooth_x is None:
                    smooth_x = raw_x
                    smooth_y = raw_y
                else:
                    smooth_x = (
                        SMOOTHING * raw_x
                        + (1 - SMOOTHING) * smooth_x
                    )

                    smooth_y = (
                        SMOOTHING * raw_y
                        + (1 - SMOOTHING) * smooth_y
                    )

                x = int(smooth_x)
                y = int(smooth_y)

                current_point = (x, y)

                # -----------------------------
                # Ignore very tiny movements
                # -----------------------------

                if previous_point is not None:

                    distance = math.hypot(
                        x - previous_point[0],
                        y - previous_point[1]
                    )

                    if distance >= MIN_MOVEMENT:

                        cv2.line(
                            canvas,
                            previous_point,
                            current_point,
                            (0, 0, 0),
                            5,
                            cv2.LINE_AA
                        )

                previous_point = current_point

                # Show fingertip
                cv2.circle(
                    frame,
                    current_point,
                    8,
                    (0, 255, 0),
                    -1
                )

            else:

                # Stop current stroke
                previous_point = None

        else:

            previous_point = None

        # -----------------------------
        # Combine webcam + canvas
        # -----------------------------

        output = cv2.addWeighted(
            frame,
            0.7,
            canvas,
            0.3,
            0
        )

        status = "DRAWING" if drawing else "NOT DRAWING"

        cv2.putText(
            output,
            status,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "AirNote - Smooth Air Drawing Test",
            output
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("✅ Smooth air drawing test finished.")