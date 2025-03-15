# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torchvision.transforms as transforms
# from gtts import gTTS
# import os
# import time
# import platform

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load models with error handling
# try:
#     # YOLOv8 model for object detection
#     yolo_model = YOLO("yolov8n.pt").to(device)
#     print("YOLOv8 model loaded successfully")
    
#     # BLIP-2 model for scene description
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
#     print("BLIP model loaded successfully")
    
#     # MiDaS model for depth estimation
#     midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
#     midas.eval()
#     midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#     print("MiDaS model loaded successfully")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     exit(1)

# # Initialize camera with error handling
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open camera")
#     exit(1)

# def capture_frame():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame!")
#         return None
#     return frame

# def detect_objects(frame):
#     try:
#         results = yolo_model(frame)
#         # Extract object names from results
#         detected_objects = []
#         for result in results:
#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 detected_objects.append(result.names[cls_id])
#         return detected_objects
#     except Exception as e:
#         print(f"Object detection error: {e}")
#         return []

# def generate_caption(frame):
#     try:
#         # Convert image to RGB and PIL format
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = transforms.ToPILImage()(image)
        
#         # Process image with BLIP
#         inputs = blip_processor(images=pil_image, return_tensors="pt").to(device)
#         with torch.no_grad():
#             caption = blip_model.generate(**inputs)
#         return blip_processor.decode(caption[0], skip_special_tokens=True)
#     except Exception as e:
#         print(f"Caption generation error: {e}")
#         return "Unable to generate caption"

# def estimate_depth(frame):
#     try:
#         # Convert to RGB and prepare for MiDaS
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_tensor = midas_transform(frame_rgb).to(device)
        
#         with torch.no_grad():
#             prediction = midas(input_tensor)
        
#         # Normalize depth map to 0-1 range
#         depth_map = prediction.squeeze().cpu().numpy()
#         depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
#         # Rough distance estimation (simplified, not true meters)
#         avg_depth = np.mean(depth_map)
#         return depth_map, avg_depth
#     except Exception as e:
#         print(f"Depth estimation error: {e}")
#         return None, 0.0

# def text_to_speech(text):
#     try:
#         tts = gTTS(text, lang="en")
#         audio_file = "output.mp3"
#         tts.save(audio_file)
        
#         # Cross-platform audio playback
#         if platform.system() == "Windows":
#             os.system(f"start {audio_file}")
#         elif platform.system() == "Darwin":  # macOS
#             os.system(f"afplay {audio_file}")
#         else:  # Linux
#             os.system(f"mpg321 {audio_file} -quiet")
        
#         # Clean up audio file
#         time.sleep(1)  # Wait for playback to finish
#         if os.path.exists(audio_file):
#             os.remove(audio_file)
#     except Exception as e:
#         print(f"Text-to-speech error: {e}")

# def main():
#     print("Starting AI-based scene description and depth estimation...")
#     time.sleep(2)

#     try:
#         while True:
#             frame = capture_frame()
#             if frame is None:
#                 break
                
#             # Object detection
#             objects = detect_objects(frame)
#             detected_text = f"Detected objects: {', '.join(objects)}." if objects else "No objects detected."
            
#             # Scene description
#             scene_description = generate_caption(frame)
            
#             # Depth estimation
#             depth_map, avg_depth = estimate_depth(frame)
#             if depth_map is None:
#                 continue
#             depth_text = f"Estimated average depth value is {avg_depth:.2f} (normalized)."
            
#             # Combine results
#             final_description = f"{scene_description}. {detected_text} {depth_text}"
#             print("AI Description:", final_description)
            
#             # Convert text to speech
#             text_to_speech(final_description)
            
#             # Display results
#             cv2.imshow("Captured Frame", frame)
#             # Convert depth map to 8-bit for display
#             depth_display = (depth_map * 255).astype(np.uint8)
#             cv2.imshow("Depth Map", depth_display)
            
#             # Exit on 'q' press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     except KeyboardInterrupt:
#         print("Program terminated by user")
#     except Exception as e:
#         print(f"Main loop error: {e}")
#     finally:
#         # Cleanup
#         cap.release()
#         cv2.destroyAllWindows()
#         if os.path.exists("output.mp3"):
#             os.remove("output.mp3")

# if __name__ == "__main__":
#     main()




import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms as transforms
from gtts import gTTS
import os
import time
import platform
import keyboard  # For listening to ` tap

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

# Text-to-speech (TTS)
def text_to_speech(text):
    try:
        tts = gTTS(text, lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)

        if platform.system() == "Windows":
            os.system(f"start {audio_file}")
        elif platform.system() == "Darwin":
            os.system(f"afplay {audio_file}")
        else:
            os.system(f"mpg321 {audio_file} -quiet")

        time.sleep(1)
        if os.path.exists(audio_file):
            os.remove(audio_file)
    except Exception as e:
        print(f"⚠ Text-to-speech error: {e}")

# Main function
def main():
    print("🎥 Starting AI-based scene description and depth estimation... (Press ``` three times to stop)")
    time.sleep(2)
    
    stop_signal = 0  # To count backtick presses
    
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
                time.sleep(0.2)  # Prevent accidental multiple detections
            else:
                stop_signal = 0  # Reset if ` is not pressed consecutively

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Program terminated by user")
    except Exception as e:
        print(f"❌ Main loop error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists("output.mp3"):
            os.remove("output.mp3")

if __name__ == "__main__":
    main()
