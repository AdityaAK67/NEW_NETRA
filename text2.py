import cv2
import easyocr
import pyttsx3
import torch
import threading

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_gpu = torch.cuda.is_available()
print(f"Using device: {device}")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech speed

# Initialize EasyOCR Readers (Use GPU if available)
reader_mr_hi = easyocr.Reader(['mr', 'hi'], gpu=use_gpu)  # Marathi, Hindi
reader_bn_as_en = easyocr.Reader(['bn', 'as', 'en'], gpu=use_gpu)  # Bengali, Assamese, English

# Cache for previous detected text
last_detected_text = ""

# Function to process the frame in a separate thread
def process_frame(frame):
    global last_detected_text
    
    # Resize frame to speed up OCR (reduce computation)
    frame_resized = cv2.resize(frame, (640, 480))  
    
    # Perform OCR
    text_mr_hi = reader_mr_hi.readtext(frame_resized)
    text_bn_as_en = reader_bn_as_en.readtext(frame_resized)

    detected_texts = []
    threshold = 0.25  # Confidence threshold

    # Process detected text
    for text_data in (text_mr_hi + text_bn_as_en):
        bbox, text, score = text_data
        if score > threshold:
            detected_texts.append(text)
            cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    # Convert detected text to speech if it's new
    if detected_texts:
        text_to_speak = " ".join(detected_texts)
        if text_to_speak != last_detected_text:  # Avoid redundant speech output
            last_detected_text = text_to_speak
            print("Detected Text:", text_to_speak)
            threading.Thread(target=speak_text, args=(text_to_speak,)).start()  # Run speech in parallel

    return frame

# Function to speak text asynchronously
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live feed
    cv2.imshow('Live Feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('z'):  # Capture frame and detect text when 'z' is pressed
        threading.Thread(target=process_frame, args=(frame.copy(),)).start()  # Process OCR in a separate thread

    elif key == ord('q'):  # Quit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
