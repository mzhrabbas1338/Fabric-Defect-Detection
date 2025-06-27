from ultralytics import YOLO
model = YOLO(r"C:\Users\SMART TECH\OneDrive - Higher Education Commission\Desktop\FDS\FabricDetectsion\yolov8n.pt")
results = model.predict(r"C:\Users\SMART TECH\OneDrive - Higher Education Commission\Desktop\FDS\FabricDetectsion\test\images\10_jpg.rf.72225263a9e2d52a57c092b24e6f4a32.jpg")
results[0].show()  # This should open a window with detections