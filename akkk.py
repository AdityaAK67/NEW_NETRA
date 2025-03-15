import cv2
import numpy as np
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Load COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Debugging information
print("Layer names:", layer_names)
print("Unconnected out layers:", net.getUnconnectedOutLayers())

# Ensure net.getUnconnectedOutLayers() returns a list of indices
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
last_announce = 0
cooldown = 5  # seconds between announcements
detected_objects = set()

def announce_objects(objects):
    if objects:
        object_list = ", ".join(objects)
        print(f"Detected: {object_list}")
        engine.say(f"Detected {object_list}")
        engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    current_objects = set()
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                label = str(classes[class_id])
                current_objects.add(label)

    # Update detected objects
    detected_objects = current_objects.copy()

    # Announce detected objects with cooldown
    if time.time() - last_announce > cooldown:
        announce_objects(detected_objects)
        last_announce = time.time()
        detected_objects.clear()

    # Display objects on screen
    cv2.putText(frame, f"Detected: {', '.join(current_objects)}", (10, 30), font, 2, (0, 255, 0), 2)
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()