from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train-3/weights/best.pt")

polygon = np.array([
    (100,100),
    (500,100),
    (500,400),
    (100,400)
], np.int32)

previous_state = {}

results = model.track(
    source = "WhatsApp Video 2026-09-12 at 2.07.50 PM.mp4",
    conf = 0.5,
    imgsz = 960,
    tracker = "bytetrack.yaml",
    persist = True,
    stream = True,
    verbose = False
)

for result in results:
    frame = result.plot()
    if result.boxes.id is None:
        continue
    
    cv2.polylines(
        frame,
        [polygon],
        True,
        (0,255,0),
        2
    )
    track_ids = result.boxes.id.int().cpu().tolist()
    boxes = result.boxes.xyxy.cpu().numpy()
    for track_id, box in zip(track_ids, boxes):
        x1, y1, x2, y2 = box
        center_x = int((x1+x2)/2)
        bottom_y = int(y2)
        point = (center_x, bottom_y)
        cv2.circle(frame,point,5,(0,0,255),-1)
        inside = cv2.pointPolygonTest(
            polygon,
            point,
            False
        ) >=0

        was_inside = previous_state.get(track_id, False)

        if not was_inside and inside:
            print("Zone Entry", track_id)
        elif was_inside and not inside:
            print("zone exit", track_id)
        previous_state[track_id] = inside
    cv2.imshow("nightguard ai - zone test",frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()