import cv2
import mediapipe as mp
import time

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
)

camera = cv2.VideoCapture(0)

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        gesture = "NO DRAW"

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            # Index finger landmarks:
            # 5  = index base
            # 6  = index middle joint
            # 7  = index upper joint
            # 8  = index fingertip

            index_tip = hand[8]
            index_joint = hand[6]

            # In the camera image, smaller Y means higher on screen.
            if index_tip.y < index_joint.y:
                gesture = "DRAW"

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "AirNote - Gesture Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()