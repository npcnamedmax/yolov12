from ultralytics import YOLO

model = YOLO('yolov12s.yaml', task='detect')  

model = model.load("YOLOv12S.pt")
# Train the model
model.train(data='data.yaml', task='detect', epochs=200, imgsz=640, patience=30, batch=8, device="cuda:0")

# Evaluate the model on the test set
# model.val(data='data.yaml', imgsz=640, device="cuda:0", split='val')
model.save("firsttry.pt")

