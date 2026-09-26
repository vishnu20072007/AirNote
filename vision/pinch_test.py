import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# MODEL
# =========================

MODEL_PATH = "vision/models/hand_landmarker.task"

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

print("✅ AirNote smooth drawing test started")
print("🤏 Pinch thumb + index to draw")
print("✋ Release pinch to stop")
print("Press Q to quit")


# =========================
# DRAWING
# =========================

strokes = []
current_stroke = []

PINCH_THRESHOLD = 45

# Smoothing strength
SMOOTHING = 0.65

# Ignore tiny movements
MIN_DISTANCE = 4

# Last smoothed point
smooth_x = None
smooth_y = None


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    pinch = False

    # =========================
    # HAND DETECTED
    # =========================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        thumb = hand[4]
        index = hand[8]

        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        index_x = int(index.x * w)
        index_y = int(index.y * h)


        # =========================
        # PINCH DISTANCE
        # =========================

        distance = math.sqrt(
            (thumb_x - index_x) ** 2 +
            (thumb_y - index_y) ** 2
        )

        if distance < PINCH_THRESHOLD:
            pinch = True


        # =========================
        # SMOOTH INDEX POSITION
        # =========================

        if smooth_x is None:

            smooth_x = index_x
            smooth_y = index_y

        else:

            smooth_x = (
                SMOOTHING * smooth_x
                + (1 - SMOOTHING) * index_x
            )

            smooth_y = (
                SMOOTHING * smooth_y
                + (1 - SMOOTHING) * index_y
            )

        smooth_x = int(smooth_x)
        smooth_y = int(smooth_y)


        # =========================
        # DRAW
        # =========================

        if pinch:

            should_add_point = True

            if current_stroke:

                last_x, last_y = current_stroke[-1]

                movement = math.sqrt(
                    (smooth_x - last_x) ** 2 +
                    (smooth_y - last_y) ** 2
                )

                # Ignore tiny movements
                if movement < MIN_DISTANCE:
                    should_add_point = False

            if should_add_point:

                current_stroke.append(
                    [smooth_x, smooth_y]
                )

            status = "PINCH / DRAW"
            text_color = (0, 255, 0)

        else:

            if current_stroke:

                strokes.append(
                    current_stroke
                )

                current_stroke = []

            status = "NO PINCH"
            text_color = (0, 0, 255)


        # =========================
        # DRAW SAVED STROKES
        # =========================

        for stroke in strokes:

            for i in range(1, len(stroke)):

                cv2.line(
                    frame,
                    tuple(stroke[i - 1]),
                    tuple(stroke[i]),
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA
                )


        # Current stroke

        for i in range(1, len(current_stroke)):

            cv2.line(
                frame,
                tuple(current_stroke[i - 1]),
                tuple(current_stroke[i]),
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )


        # =========================
        # LANDMARKS
        # =========================

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # =========================
        # THUMB
        # =========================

        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            10,
            (255, 0, 0),
            -1
        )


        # =========================
        # SMOOTH INDEX
        # =========================

        cv2.circle(
            frame,
            (smooth_x, smooth_y),
            10,
            (0, 255, 255),
            -1
        )


        # =========================
        # THUMB ↔ INDEX
        # =========================

        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            text_color,
            2
        )


        # =========================
        # INFO
        # =========================

        cv2.putText(
            frame,
            f"Distance: {int(distance)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2
        )


    else:

        if current_stroke:

            strokes.append(
                current_stroke
            )

            current_stroke = []

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
    # SHOW
    # =========================

    cv2.imshow(
        "AirNote - Smooth Pinch Drawing",
        frame
    )


    # =========================
    # QUIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# SAVE LAST STROKE
# =========================

if current_stroke:

    strokes.append(
        current_stroke
    )


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
detector.close()


print("✅ Smooth drawing test stopped")
print(
    f"Total strokes captured: {len(strokes)}"
)

for i, stroke in enumerate(strokes):

    print(
        f"Stroke {i + 1}: "
        f"{len(stroke)} points"
    )