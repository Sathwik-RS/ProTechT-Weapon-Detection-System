from ultralytics import YOLO
import cv2

# Load a YOLOv8 model
model = YOLO("best3.pt")  # Replace with 'yolov8n.pt' or custom gun model

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLOv8 inference
    results = model.predict(frame, imgsz=640, conf=0.4)

    # Visualize detections
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("YOLOv8 Gun Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
