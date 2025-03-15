    
import cv2
import requests
import pyttsx3
import time
import os
from threading import Thread
from queue import Queue
import numpy as np

# Constants
OCR_API_KEY = "K88814121588957"
OCR_API_URL = "https://api.ocr.space/parse/image"
IMAGE_PATH = "captured_text.jpg"

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Queue for non-blocking TTS
tts_queue = Queue()

def text_to_speech_worker():
    """Background thread for non-blocking text-to-speech."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")
        tts_queue.task_done()

Thread(target=text_to_speech_worker, daemon=True).start()

def initialize_camera():
    """Initialize camera with brighter settings."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return None
    
    # Optimized resolution and bright settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    cap.set(cv2.CAP_PROP_EXPOSURE, -2)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print(f"[INFO] Camera resolution: {width}x{height}, Brightness: {brightness}")
    return cap

def preprocess_image(frame):
    """Preprocess image for brightness and OCR accuracy."""
    # Brighten image
    brightened = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    # Convert to grayscale
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    # Light noise reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive thresholding for text clarity
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def capture_image(cap):
    """Captures image with stabilization and preprocessing."""
    for _ in range(3):
        cap.read()
    
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image!")
        return None

    processed_frame = preprocess_image(frame)
    cv2.imwrite(IMAGE_PATH, processed_frame)
    cv2.imwrite("original_capture.jpg", frame)
    return IMAGE_PATH

def ocr_space_api(image_path):
    """OCR.Space API call with enhanced error handling."""
    if not os.path.exists(image_path):
        return "Error: Image file not found."

    try:
        with open(image_path, 'rb') as image_file:
            response = requests.post(
                OCR_API_URL,
                files={'filename': image_file},
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'eng',
                    'isOverlayRequired': True,
                    'OCREngine': 2
                },
                timeout=10,
                headers={'User-Agent': 'OCRTextScanner/1.0'}
            )
        
        result = response.json()
        if response.status_code != 200:
            return f"API Error: Status {response.status_code}"
        if result.get("ParsedResults"):
            text = result["ParsedResults"][0]["ParsedText"].strip()
            return text if text else "No text detected."
        return f"OCR failed: {result.get('ErrorMessage', 'Unknown error')}"
    except requests.Timeout:
        return "Error: OCR API timeout"
    except Exception as e:
        return f"Error in OCR API: {str(e)}"

def main():
    cap = initialize_camera()
    if not cap:
        return

    print("[INFO] Press 's' to scan text, 'q' to quit, 'b' to adjust brightness.")
    last_scan_time = 0
    cooldown = 1.5
    brightness_increment = 150

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed!")
                break

            # Minimal overlay without green box
            h, w = frame.shape[:2]
            cv2.putText(frame, f"s: Scan, q: Quit, b: Brightness ({brightness_increment})", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Text Scanner", frame)

            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()

            if key == ord('s') and (current_time - last_scan_time) > cooldown:
                image_path = capture_image(cap)
                if image_path:
                    text = ocr_space_api(image_path)
                    print(f"[INFO] Detected: {text}")
                    tts_queue.put(text)
                    last_scan_time = current_time
            elif key == ord('b'):
                brightness_increment = min(255, brightness_increment + 20)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_increment)
                print(f"[INFO] Brightness set to {brightness_increment}")
            elif key == ord('q'):
                print("[INFO] Exiting program.")
                break

    finally:
        tts_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()
        for file in [IMAGE_PATH, "original_capture.jpg"]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    main()    