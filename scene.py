import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms as transforms
import pyttsx3
import time
import keyboard

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models with error handling
try:
    yolo_model = YOLO("yolov8n.pt").to(device)
    print("✅ YOLOv8 model loaded successfully")

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    print("✅ BLIP model loaded successfully")

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
    midas.eval()
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    print("✅ MiDaS model loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit(1)

# Initialize pyttsx3 with explicit engine check
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)  # Use default voice
    print("✅ pyttsx3 initialized successfully")
except Exception as e:
    print(f"❌ Error initializing pyttsx3: {e}")
    exit(1)

# Function to capture frames
def capture_frame():
    ret, frame = cap.read()
    return frame if ret else None

# Object detection
def detect_objects(frame):
    try:
        results = yolo_model(frame)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detected_objects.append(result.names[cls_id])
        return detected_objects
    except Exception as e:
        print(f"⚠ Object detection error: {e}")
        return []

# Generate scene caption
def generate_caption(frame):
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = transforms.ToPILImage()(image)
        inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            caption = blip_model.generate(**inputs)
        return blip_processor.decode(caption[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠ Caption generation error: {e}")
        return "Unable to generate caption"

# Depth estimation
def estimate_depth(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transform(frame_rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_tensor)

        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        avg_depth = np.mean(depth_map)
        return depth_map, avg_depth
    except Exception as e:
        print(f"⚠ Depth estimation error: {e}")
        return None, 0.0

# Text-to-speech with pyttsx3
def text_to_speech(text):
    try:
        print(f"🔊 Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"⚠ Text-to-speech error: {e}")
        # Fallback test to confirm audio output
        try:
            engine.say("Test audio output")
            engine.runAndWait()
            print("ℹ Fallback test audio played")
        except:
            print("❌ pyttsx3 failed completely")

# Main function
def main():
    print("🎥 Starting AI-based scene description and depth estimation... (Press ``` three times to stop)")
    time.sleep(2)
    
    stop_signal = 0
    
    try:
        while True:
            frame = capture_frame()
            if frame is None:
                break

            objects = detect_objects(frame)
            detected_text = f"Detected objects: {', '.join(objects)}." if objects else "No objects detected."
            scene_description = generate_caption(frame)
            depth_map, avg_depth = estimate_depth(frame)
            depth_text = f"Estimated average depth value is {avg_depth:.2f}."

            final_description = f"{scene_description}. {detected_text} {depth_text}"
            print("🔊 AI Description:", final_description)

            text_to_speech(final_description)

            cv2.imshow("Captured Frame", frame)
            depth_display = (depth_map * 255).astype(np.uint8)
            cv2.imshow("Depth Map", depth_display)

            if keyboard.is_pressed('`'):
                stop_signal += 1
                if stop_signal >= 3:
                    print("🛑 Stopping program...")
                    break
                time.sleep(0.2)
            else:
                stop_signal = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Program terminated by user")
    except Exception as e:
        print(f"❌ Main loop error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()