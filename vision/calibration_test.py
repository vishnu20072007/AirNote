import cv2
import json
import os

# ============================================================
# AIRNOTE - SCREEN CALIBRATION
# ============================================================

WINDOW_NAME = "AirNote - Screen Calibration"

points = []

# ------------------------------------------------------------
# MOUSE CLICK
# ------------------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append([x, y])

            print(
                f"Point {len(points)}: "
                f"({x}, {y})"
            )


# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

if not camera.isOpened():

    print("ERROR: Webcam could not be opened.")
    raise SystemExit


# ------------------------------------------------------------
# WINDOW
# ------------------------------------------------------------

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


print()
print("==============================================")
print("      AIRNOTE SCREEN CALIBRATION")
print("==============================================")
print()
print("IMPORTANT:")
print("Make sure the FULL laptop screen is visible")
print("inside the camera.")
print()
print("Click the ACTUAL LAPTOP SCREEN corners.")
print()
print("Order:")
print("1. TOP LEFT")
print("2. TOP RIGHT")
print("3. BOTTOM RIGHT")
print("4. BOTTOM LEFT")
print()
print("Do NOT click the camera window corners.")
print("Do NOT click the black background.")
print()
print("Press R to restart.")
print("Press ENTER after 4 points.")
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        break


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    display = frame.copy()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    cv2.putText(
        display,
        "CLICK LAPTOP SCREEN 4 CORNERS",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    if len(points) == 0:

        text = "Click TOP LEFT"

    elif len(points) == 1:

        text = "Click TOP RIGHT"

    elif len(points) == 2:

        text = "Click BOTTOM RIGHT"

    elif len(points) == 3:

        text = "Click BOTTOM LEFT"

    else:

        text = "Press ENTER to save"


    cv2.putText(
        display,
        text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # DRAW SELECTED POINTS
    # --------------------------------------------------------

    for i, point in enumerate(points):

        x, y = point

        cv2.circle(
            display,
            (x, y),
            9,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            str(i + 1),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # CONNECT POINTS
    # --------------------------------------------------------

    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(
                display,
                tuple(points[i]),
                tuple(points[i + 1]),
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )


    if len(points) == 4:

        cv2.line(
            display,
            tuple(points[3]),
            tuple(points[0]),
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # RESTART
    # --------------------------------------------------------

    if key == ord("r"):

        points = []

        print()
        print("Calibration restarted.")
        print("Click TOP LEFT again.")


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elif key == 13:

        if len(points) == 4:

            calibration = {

                "camera_width": int(
                    frame.shape[1]
                ),

                "camera_height": int(
                    frame.shape[0]
                ),

                "points": {

                    "top_left": points[0],

                    "top_right": points[1],

                    "bottom_right": points[2],

                    "bottom_left": points[3]
                }
            }


            path = (
                "vision/calibration.json"
            )


            os.makedirs(
                "vision",
                exist_ok=True
            )


            with open(
                path,
                "w"
            ) as f:

                json.dump(
                    calibration,
                    f,
                    indent=4
                )


            print()
            print("==============================================")
            print("CALIBRATION SAVED")
            print("==============================================")
            print(
                json.dumps(
                    calibration,
                    indent=4
                )
            )
            print()
            print(
                "Saved to:",
                path
            )


            break


        else:

            print()
            print(
                "Need 4 points."
            )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        print(
            "Calibration cancelled."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()