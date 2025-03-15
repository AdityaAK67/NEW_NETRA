import os
import cv2
import time
import pyttsx3
from voice_commands import recognize_voice
from object_detection import detect_objects
from text_detection import detect_text
from hazard_detection import detect_hazards

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Initialize Camera
cap = cv2.VideoCapture(0)

# Function to Provide Voice Feedback
def speak(text):
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function for Automatic Mode Switching
def auto_switch_mode(frame):
    objects = detect_objects(frame)
    text = detect_text(frame)
    hazards = detect_hazards(frame)

    if hazards:
        return "hazard detection"
    elif text:
        return "text detection"
    elif objects:
        return "object detection"
    else:
        return "idle"

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Auto Mode Switching
    mode = auto_switch_mode(frame)
    print(f"Switching to mode: {mode}")
    speak(f"Switching to {mode} mode.")

    # Voice Command Handling
    command = recognize_voice()
    if command:
        if "read" in command:
            text = detect_text(frame)
            speak(f"Detected text: {text}" if text else "No text found.")
        
        elif "what's ahead" in command:
            objects = detect_objects(frame)
            speak(f"I see {', '.join(map(str, objects))} ahead." if objects else "No objects detected.")

        elif "hazard alert" in command:
            hazards = detect_hazards(frame)
            speak(f"Warning: {', '.join(hazards)} detected." if hazards else "No hazards detected.")

        elif "exit" in command:
            speak("Shutting down.")
            break

cap.release()
cv2.destroyAllWindows()
