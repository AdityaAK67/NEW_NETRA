import cv2
from ultralytics import YOLO

# Load the YOLO model (ensure it runs on CPU)
model = YOLO("yolov8n.pt")  # You can use different YOLOv8 models, such as 'yolov8n.pt', 'yolov8s.pt', etc.

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        continue  # Skip frame

    # Run inference
    results = model(frame)

    # Process the results (detections)
    for result in results[0].boxes:  # Loop through the boxes
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordinates of the bounding box
        confidence = result.conf[0].item()  # Confidence score
        class_id = int(result.cls[0].item())  # Class ID
        label = model.names[class_id]  # Class label

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
