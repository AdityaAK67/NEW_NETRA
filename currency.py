import cv2
import torch
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
import time

# Load YOLOv8 model trained for currency detection
model = YOLO("yolov8n.pt")  # Use a trained YOLO model for currency

# Initialize camera
cap = cv2.VideoCapture(0)  # Change index if using Raspberry Pi camera

def detect_currency(frame):
    results = model(frame)  # Run YOLO on frame
    detected_notes = []

    for result in results:
        for cls in result.boxes.cls:
            detected_notes.append(result.names[int(cls)])  # Get currency label

    return detected_notes

def text_to_speech(text):
    tts = gTTS(text, lang="en")
    tts.save("currency.mp3")
    os.system("mpg321 currency.mp3")  # For Linux/Raspberry Pi, use 'afplay' on macOS

def main():
    print("Starting Currency Detection...")
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            continue

        # Detect currency
        notes = detect_currency(frame)
        detected_text = f"Detected currency: {', '.join(notes)}" if notes else "No currency detected."

        # Convert text to speech
        print(detected_text)
        text_to_speech(detected_text)

        # Display the frame
        cv2.imshow("Currency Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
