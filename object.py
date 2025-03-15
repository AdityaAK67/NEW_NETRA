
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import warnings
warnings.filterwarnings('ignore')  # Ignore Python warnings

from absl import logging
logging.set_verbosity(logging.ERROR)  # Suppress absl warnings

from ultralytics import YOLO
import cv2
import pyttsx3
import mediapipe as mp
import threading
from queue import Queue
import time
from collections import Counter

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

announcement_queue = Queue()

def announce():
    while True:
        text = announcement_queue.get()
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=announce, daemon=True).start()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# YOLO model with correct class names for YOLOv8n
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame_skip = 2
frame_count = 0
last_person_announcement_time = time.time()
last_object_announcement_time = time.time()

def detect_activity(landmarks):
    try:
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        if left_ankle.y > left_knee.y and right_ankle.y > right_knee.y:
            return "Sitting"
        if abs(left_hip.y - right_hip.y) < 0.1 and abs(left_knee.y - right_knee.y) < 0.1:
            return "Lying Down"
        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            return "Raising Hands"
        return "Standing"
    except:
        return "Idle"

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # YOLO detection
    results = model.predict(img, conf=0.5, iou=0.4, verbose=False)  # Disable YOLO verbose
    detected_objects = Counter()
    person_activities = []

    for r in results:
        for box in r.boxes:
            if box.conf[0] < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            if cls >= len(classNames):
                continue  # Skip invalid class indices
            
            class_name = classNames[cls]
            detected_objects[class_name] += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"{class_name} {box.conf[0]*100:.1f}%",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            if class_name == "person":
                person_img = img[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                results_pose = pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                if results_pose.pose_landmarks:
                    person_activities.append(detect_activity(results_pose.pose_landmarks.landmark))

    current_time = time.time()
    if current_time - last_object_announcement_time >= 3:
        for obj, count in detected_objects.items():
            if obj != "person":
                announcement_queue.put(f"{obj} detected {count} time(s).")
        last_object_announcement_time = current_time

    if current_time - last_person_announcement_time >= 10:
        if person_activities:
            most_common = Counter(person_activities).most_common(1)[0][0]
            announcement_queue.put(f"Person is {most_common}.")
        last_person_announcement_time = current_time

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
