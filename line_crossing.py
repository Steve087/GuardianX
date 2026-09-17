from ultralytics import YOLO
import cv2
from event_manager import create_event


# ============================================================
# MODEL
# ============================================================

model = YOLO("runs/detect/train-3/weights/best.pt")


# ============================================================
# LINE STATE
# ============================================================

line_points = []
line_confirmed = False


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global line_points

    if event == cv2.EVENT_LBUTTONDOWN:

        # Only two points are needed for a line
        if len(line_points) < 2:

            line_points.append((x, y))


# ============================================================
# WINDOW
# ============================================================

window_name = "NightGuard AI - Line Crossing"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


# ============================================================
# VIDEO
# ============================================================

video_path = "video6125123190915079860.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:

    print("Could not read video")

    cap.release()
    cv2.destroyAllWindows()

    exit()


# ============================================================
# LINE SELECTION
# ============================================================

while True:

    display_frame = frame.copy()


    # --------------------------------------------------------
    # DRAW SELECTED POINTS
    # --------------------------------------------------------

    for point in line_points:

        cv2.circle(
            display_frame,
            point,
            5,
            (0, 0, 255),
            -1
        )


    # --------------------------------------------------------
    # DRAW LINE
    # --------------------------------------------------------

    if len(line_points) == 2:

        cv2.line(
            display_frame,
            line_points[0],
            line_points[1],
            (255, 0, 0),
            3
        )


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    if not line_confirmed:

        text = "Click 2 points | C = confirm | Q = quit"

    else:

        text = "Line confirmed | Q = quit"


    cv2.putText(
        display_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        window_name,
        display_frame
    )


    key = cv2.waitKey(30) & 0xFF


    # --------------------------------------------------------
    # CONFIRM LINE
    # --------------------------------------------------------

    if key == ord("c"):

        if len(line_points) == 2:

            line_confirmed = True

            print("Line confirmed")
            print("Start:", line_points[0])
            print("End:", line_points[1])


    # --------------------------------------------------------
    # START TRACKING AFTER CONFIRMATION
    # --------------------------------------------------------

    if line_confirmed:

        break


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()

        exit()


# ============================================================
# SAVE LINE COORDINATES
# ============================================================

line_start = line_points[0]
line_end = line_points[1]


# ============================================================
# BYTE TRACK
# ============================================================

results = model.track(

    source=video_path,

    conf=0.5,

    imgsz=960,

    tracker="bytetrack.yaml",

    persist=True,

    stream=True,

    verbose=False
)


# ============================================================
# PROCESS TRACKING RESULTS
# ============================================================
previous_side = {}
crossing_cooldown = {}
frame_number = 0
cooldown_frames = 30

for result in results:

    frame = result.plot()
    frame_number+=1


    # --------------------------------------------------------
    # DRAW USER-DEFINED LINE
    # --------------------------------------------------------

    cv2.line(
        frame,
        line_start,
        line_end,
        (255, 0, 0),
        3
    )


    # --------------------------------------------------------
    # CHECK WHETHER TRACK IDs EXIST
    # --------------------------------------------------------

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )


        # Bounding boxes

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # PROCESS EACH TRACK
        # ----------------------------------------------------

        for track_id, box, confidence in zip(
            track_ids,
            boxes,
            confidences
        ):

            x1, y1, x2, y2 = box


            # ------------------------------------------------
            # BOTTOM-CENTER POINT
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            bottom_y = int(y2)


            point = (
                center_x,
                bottom_y
            )

            side = (
                (line_end[0] - line_start[0]) * (point[1] - line_start[1])
                -
                (line_end[1] - line_start[1]) * (point[0] - line_start[0])
            )

            

            # calculate side

            if side > 0:
                current_side = 1

            elif side < 0:
                current_side = -1

            else:
                current_side = 0

            previous = previous_side.get(track_id)

        last_crossing = crossing_cooldown.get(track_id)

        if current_side != 0:

            if previous is not None:

                if previous == 1 and current_side == -1:

                    if (
                        last_crossing is None
                        or frame_number - last_crossing > cooldown_frames
                    ):

                        event = create_event(
                        event_type="line_crossing",
                        track_id=track_id,
                        confidence=float(confidence),
                        direction="A -> B"
                    )

                    print(event)

                    crossing_cooldown[track_id] = frame_number


                elif previous == -1 and current_side == 1:

                    if (
                        last_crossing is None
                        or frame_number - last_crossing > cooldown_frames
                    ):

                        event = create_event(
                        event_type="line_crossing",
                        track_id=track_id,
                        confidence=float(confidence),
                        direction="B-> A"
                    )

                        print(event)

                        crossing_cooldown[track_id] = frame_number

            previous_side[track_id] = current_side


            # ------------------------------------------------
            # DRAW BOTTOM-CENTER POINT
            # ------------------------------------------------

            cv2.circle(
                frame,
                point,
                5,
                (0, 0, 255),
                -1
            )


            # ------------------------------------------------
            # DISPLAY TRACK ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        window_name,
        frame
    )


    key = cv2.waitKey(30) & 0xFF


    if key == ord("q"):

        break



# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()