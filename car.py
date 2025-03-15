# # import cv2
# # import torch
# # import numpy as np
# # import time
# # import os
# # import platform
# # import RPi.GPIO as GPIO
# # from ultralytics import YOLO
# # from gtts import gTTS

# # # 🚀 Load YOLOv8 model for real-time object detection
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # yolo_model = YOLO("yolov8n.pt").to(device)
# # print("✅ YOLOv8 model loaded successfully!")

# # # 🚦 GPIO Setup for Ultrasonic Sensor (For Raspberry Pi)
# # TRIG = 23
# # ECHO = 24
# # GPIO.setmode(GPIO.BCM)
# # GPIO.setup(TRIG, GPIO.OUT)
# # GPIO.setup(ECHO, GPIO.IN)

# # def get_distance():
# #     """Measure distance using Ultrasonic Sensor (HC-SR04)"""
# #     GPIO.output(TRIG, True)
# #     time.sleep(0.00001)
# #     GPIO.output(TRIG, False)

# #     start_time, stop_time = time.time(), time.time()

# #     while GPIO.input(ECHO) == 0:
# #         start_time = time.time()
# #     while GPIO.input(ECHO) == 1:
# #         stop_time = time.time()

# #     elapsed_time = stop_time - start_time
# #     distance = (elapsed_time * 34300) / 2  # Speed of sound: 34300 cm/s
# #     return round(distance, 2)

# # def text_to_speech(text):
# #     """Convert text to speech"""
# #     try:
# #         tts = gTTS(text, lang="en")
# #         tts.save("alert.mp3")

# #         if platform.system() == "Windows":
# #             os.system("start alert.mp3")
# #         elif platform.system() == "Darwin":
# #             os.system("afplay alert.mp3")
# #         else:
# #             os.system("mpg321 alert.mp3 -quiet")

# #         time.sleep(1)
# #         os.remove("alert.mp3")
# #     except Exception as e:
# #         print(f"⚠ TTS Error: {e}")

# # # 🎥 Initialize Camera
# # cap = cv2.VideoCapture(0)
# # if not cap.isOpened():
# #     print("❌ Error: Could not open camera")
# #     exit(1)

# # def detect_objects(frame):
# #     """Detect vehicles and traffic lights using YOLOv8"""
# #     try:
# #         results = yolo_model(frame)
# #         detected_objects = []

# #         for result in results:
# #             for box in result.boxes:
# #                 cls_id = int(box.cls[0])
# #                 label = result.names[cls_id]
# #                 detected_objects.append(label)

# #         return detected_objects
# #     except Exception as e:
# #         print(f"⚠ Object Detection Error: {e}")
# #         return []

# # def main():
# #     print("🚀 Traffic & Pedestrian Assistance System Started! (Press 'q' to quit)")

# #     try:
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 print("❌ Error: Frame capture failed")
# #                 break

# #             detected_objects = detect_objects(frame)
# #             distance = get_distance()
# #             safe_to_cross = False

# #             if "car" in detected_objects or "truck" in detected_objects or "bus" in detected_objects:
# #                 if distance < 200:  # If vehicle is within 2 meters
# #                     alert_text = "⚠ Warning! Vehicle detected nearby. Do not cross!"
# #                     safe_to_cross = False
# #                 else:
# #                     alert_text = "Vehicle detected but at a safe distance."
# #                     safe_to_cross = True
# #             elif "traffic light" in detected_objects:
# #                 alert_text = "Traffic light detected. Wait for pedestrian signal!"
# #                 safe_to_cross = False
# #             elif "person" in detected_objects and safe_to_cross:
# #                 alert_text = "✅ Safe to cross. No vehicles nearby."
# #             else:
# #                 alert_text = "🔎 No relevant objects detected."

# #             print(f"🔊 {alert_text}")
# #             text_to_speech(alert_text)

# #             # Display results
# #             cv2.imshow("Traffic & Pedestrian Assistance", frame)

# #             # Exit on 'q' key press
# #             if cv2.waitKey(1) & 0xFF == ord("q"):
# #                 break

# #     except KeyboardInterrupt:
# #         print("🛑 System stopped by user.")
# #     except Exception as e:
# #         print(f"❌ Error: {e}")
# #     finally:
# #         cap.release()
# #         cv2.destroyAllWindows()
# #         GPIO.cleanup()

# # if __name__ == "__main__":
# #     main()


# import cv2
# import torch
# import numpy as np
# import time
# import os
# from ultralytics import YOLO
# from gtts import gTTS
# from playsound import playsound

# # Load YOLOv8 model
# model = YOLO("yolov8n.pt")  # Using pre-trained YOLOv8 model

# # Initialize webcam
# cap = cv2.VideoCapture(0)  # Change to 1 if using external webcam

# # Class IDs for vehicles & traffic signals
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# TRAFFIC_LIGHT_CLASS = 9  # Traffic light

# # Function to generate voice alerts
# def speak(text):
#     tts = gTTS(text=text, lang="en")
#     tts.save("alert.mp3")
#     playsound("alert.mp3")
#     os.remove("alert.mp3")  # Clean up

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect objects
#     results = model(frame)

#     # Initialize alert flags
#     vehicle_detected = False
#     red_light_detected = False

#     # Process detections
#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls[0])  # Class ID
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

#             # Draw bounding boxes
#             color = (0, 255, 0)  # Green for general objects
#             label = f"{model.names[cls]}"

#             if cls in VEHICLE_CLASSES:
#                 color = (0, 0, 255)  # Red for vehicles
#                 vehicle_detected = True

#             if cls == TRAFFIC_LIGHT_CLASS:
#                 color = (255, 0, 0)  # Blue for traffic lights
#                 red_light_detected = True

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Trigger alerts
#     if vehicle_detected:
#         speak("Warning! Vehicle approaching. Please wait.")

#     if red_light_detected:
#         speak("Stop! Red light detected.")

#     # Display output
#     cv2.imshow("Traffic & Pedestrian Assistance", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import torch
import numpy as np
import time
import os
from ultralytics import YOLO
from gtts import gTTS
import pygame  # Use pygame for stable audio playback

# Initialize pygame mixer
pygame.mixer.init()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
TRAFFIC_LIGHT_CLASS = 9  # Traffic light

# Function to generate and play voice alerts
def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "alert.mp3"
    tts.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.5)

    pygame.mixer.music.unload()  # Ensure file is released
    os.remove(filename)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    vehicle_detected = False
    red_light_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0)
            label = f"{model.names[cls]}"

            if cls in VEHICLE_CLASSES:
                color = (0, 0, 255)
                vehicle_detected = True

            if cls == TRAFFIC_LIGHT_CLASS:
                color = (255, 0, 0)
                red_light_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if vehicle_detected:
        speak("Warning! Vehicle approaching. Please wait.")

    if red_light_detected:
        speak("Stop! Red light detected.")

    cv2.imshow("Traffic & Pedestrian Assistance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
