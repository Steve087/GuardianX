from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.info()

if __name__ == "__main__":
    model.val(data="data.yaml",device=0)