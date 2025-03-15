from ultralytics import YOLO
import cv2
import pyttsx3
import mediapipe as mp
import threading
from queue import Queue
import time
from collections import Counter

# Text-to-Speech (TTS) Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

announcement_queue = Queue()

def announce():
    while True:
        text = announcement_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()

threading.Thread(target=announce, daemon=True).start()

# Threaded Webcam Capture
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, 640)  # Width
        self.stream.set(4, 480)  # Height
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Lazy Load YOLO Model for Faster Startup
model = None

def load_yolo():
    global model
    if model is None:
        model = YOLO("yolo-Weights/yolov8n.pt")  # Load YOLO only when needed

# Custom Object Names (If Not in Default YOLO)
custom_objects = ["fan", "zebra crossing", "traffic light", "pen", "bump", "stone", "tree", "bench", "chair", "table"]

# MediaPipe Pose Model (Optimized)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame_skip = 5  # Process every 5th frame for better speed
frame_count = 0
last_person_announcement_time = time.time()
last_object_announcement_time = time.time()

# Object Detection Function
def object_detection(img):
    """Performs object detection using YOLO model."""
    load_yolo()  # Load YOLO only on first use
    results = model.predict(img, conf=0.5, iou=0.4, verbose=False)  # Disable verbose for speed
    detected_objects = Counter()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls] if cls < len(model.names) else "Unknown"
            
            # Add Custom Objects (Manually if not in YOLO)
            if class_name == "Unknown" and cls < len(custom_objects):
                class_name = custom_objects[cls]
            
            detected_objects[class_name] += 1

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return detected_objects, img

# Detect Activity from Pose
def detect_activity(landmarks):
    """Detects a person's activity based on keypoints."""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if left_ankle.y > left_hip.y and right_ankle.y > right_hip.y:
        return "Sitting"
    elif abs(left_hip.y - right_hip.y) < 0.1:
        return "Standing"
    return "Idle"

# Start Video Stream
vs = VideoStream()

while True:
    frame_count += 1
    success, img = vs.read()
    if not success:
        break

    # Process only every 5th frame for speed
    if frame_count % frame_skip == 0:
        detected_objects, img = object_detection(img)

        # Announce objects every 5 seconds
        if time.time() - last_object_announcement_time >= 5:
            for obj in detected_objects:
                if obj != "person":
                    announcement_queue.put(f"{obj} detected {detected_objects[obj]} time(s).")
            last_object_announcement_time = time.time()

        # Pose detection for persons
        person_activities = []
        if "person" in detected_objects:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_img)
            if results_pose.pose_landmarks:
                action = detect_activity(results_pose.pose_landmarks.landmark)
                person_activities.append(action)

        # Announce person activities every 10 seconds
        if time.time() - last_person_announcement_time >= 10:
            if person_activities:
                for activity in set(person_activities):
                    announcement_queue.put(f"Person is {activity}.")
            else:
                announcement_queue.put("Person is idle.")
            last_person_announcement_time = time.time()

    # Display video feed
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
