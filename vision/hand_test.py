import cv2
import mediapipe as mp
import time

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure the hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="vision/models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Hand tracking started!")
print("Press Q to close.")

start_time = time.time()
frame_count = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            print("❌ Could not read webcam frame")
            break

        # Mirror the webcam
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Create increasing timestamp
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # Detect hands
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Draw detected landmarks
        if results.hand_landmarks:
            for hand in results.hand_landmarks:

                # Draw points
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17), (0, 17)
                ]

                for start, end in connections:
                    x1 = int(hand[start].x * frame.shape[1])
                    y1 = int(hand[start].y * frame.shape[0])

                    x2 = int(hand[end].x * frame.shape[1])
                    y2 = int(hand[end].y * frame.shape[0])

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        cv2.imshow(
            "AirNote - Hand Tracking Test",
            frame
        )

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("✅ Hand tracking test finished.")