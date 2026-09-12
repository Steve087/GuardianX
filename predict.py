from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train-3/weights/best.pt")

polygon_points = []
polygon_closed = False

def mouse_callback(event,x,y,flags,param):
    global polygon_points, polygon_closed

    if event == cv2.EVENT_LBUTTONDOWN and not polygon_closed:
        polygon_points.append((x,y))

previous_state = {}

window_name = "nightguard ai - zone test"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name,mouse_callback)

results = model.track(
    source = "video6125123190915079860.mp4",
    conf = 0.5,
    imgsz = 960,
    tracker = "bytetrack.yaml",
    persist = True,
    stream = True,
    verbose = False
)

for result in results:
    frame = result.plot()
    
    
    if len(polygon_points) >=2:
        cv2.polylines(
            frame,
            [np.array(polygon_points, np.int32)],
            polygon_closed,
            (0,255,0),
            2
        )
    if result.boxes.id is None:
        continue
    track_ids = result.boxes.id.int().cpu().tolist()
    boxes = result.boxes.xyxy.cpu().numpy()
    for track_id, box in zip(track_ids, boxes):
        x1, y1, x2, y2 = box
        center_x = int((x1+x2)/2)
        bottom_y = int(y2)
        point = (center_x, bottom_y)
        cv2.circle(frame,point,5,(0,0,255),-1)
        
        if polygon_closed == True:
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
    key = cv2.waitKey(100) & 0xFF
    
    if key == ord("c") and len(polygon_points) >= 3:
        polygon_closed = True
        polygon = np.array(polygon_points,np.int32)
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()