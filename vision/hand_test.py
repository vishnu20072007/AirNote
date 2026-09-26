import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# MODEL PATH
# =========================

MODEL_PATH = "vision/models/hand_landmarker.task"


# =========================
# MEDIAPIPE HAND LANDMARKER
# =========================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera started")
print("✋ Show your hand")
print("Press Q to quit")


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Failed to read camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    result = detector.detect(mp_image)


    # =========================
    # HAND DETECTED
    # =========================

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # Draw all landmarks
            for landmark in hand_landmarks:

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # =========================
            # INDEX FINGER TIP
            # Landmark 8
            # =========================

            index_tip = hand_landmarks[8]

            h, w, _ = frame.shape

            index_x = int(
                index_tip.x * w
            )

            index_y = int(
                index_tip.y * h
            )


            # Highlight index finger
            cv2.circle(
                frame,
                (index_x, index_y),
                12,
                (0, 255, 255),
                -1
            )


            # Show coordinates
            cv2.putText(
                frame,
                f"Index: ({index_x}, {index_y})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            # Hand detected text
            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =========================
    # SHOW CAMERA
    # =========================

    cv2.imshow(
        "AirNote - Hand Tracking",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("Camera stopped.")