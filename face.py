
import os
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading

# Create folder for detected faces
folder_path = "detected_faces"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Load Face and Eye Detection Models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Face Recognizer (LBPH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load stored faces
face_database = {}
images, labels = [], []
label_map = {}

for idx, file in enumerate(os.listdir(folder_path)):
    name, ext = os.path.splitext(file)
    if ext == ".jpg":
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(idx)
        label_map[idx] = name
        face_database[name] = img_path

# Train Face Recognizer
if images:
    face_recognizer.train(images, np.array(labels))

# Function to recognize face asynchronously
def recognize_face(face_roi):
    try:
        label, confidence = face_recognizer.predict(face_roi)
        if confidence < 50:  # Adjust threshold if needed
            person_name = label_map[label]
            engine.say(f"Hello {person_name}!")
            engine.runAndWait()
            return person_name
    except:
        return None
    return None

# Function to take voice input for new faces
def get_voice_name():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Listening for name...")
            audio = recognizer.listen(source, timeout=5)
            name = recognizer.recognize_google(audio)
            return name
        except:
            return None

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set higher FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recognize face in a separate thread
        thread = threading.Thread(target=recognize_face, args=(face_roi,))
        thread.start()
        thread.join()  # Wait for recognition to complete

        # If face is new, ask for name and save
        if recognize_face(face_roi) is None:
            engine.say("New face detected. What is your name?")
            engine.runAndWait()
            
            name = get_voice_name()
            if name:
                file_path = os.path.join(folder_path, f"{name}.jpg")
                cv2.imwrite(file_path, face_roi)
                images.append(face_roi)
                labels.append(len(label_map))
                label_map[len(label_map)] = name
                face_database[name] = file_path

                face_recognizer.train(images, np.array(labels))  # Retrain model

                engine.say(f"Face stored as {name}")
                engine.runAndWait()

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
