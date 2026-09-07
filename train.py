from ultralytics import YOLO
model = YOLO("yolo26n.pt")

if __name__ == "__main__":
    model.train(data="data.yaml", epochs=50, batch=-1, imgsz=640, device=0, optimizer="AdamW", lr0=0.001, patience=20, mosaic =0.5, mixup=0.0, copy_paste = 0.0, seed=42)