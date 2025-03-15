# import cv2
# import supervision as sv
# import queue
# import threading
# import pyttsx3
# import time
# import numpy as np
# from inference.models.yolo_world.yolo_world import YOLOWorld

# # Initialize YOLO World model
# model = YOLOWorld(model_id="yolo_world/s")  # Use the smaller YOLOWorld model for faster processing
# classes = [
#     # People-related
#     "person", "man", "woman", "child", "baby", "adult", "teenager", "elderly", "doctor", "nurse", "teacher", "student",
    
#     # Household items
#     "backpack", "chair", "table", "sofa", "bed", "tv", "remote", "microwave", "fridge", "oven", "fan", "toaster", 
#     "dishwasher", "light", "lamp", "clock", "calendar", "bookshelf", "keyboard", "mouse", "computer", "laptop", 
#     "phone", "tablet", "speaker", "camera", "headphones", "watch", "wallet", "glasses", "trashcan", "recycle bin", 
#     "scissors", "plate", "fork", "knife", "spoon", "cup", "mug", "bottle", "glass", "napkin", "towel", "toothbrush", 
#     "toothpaste", "comb", "razor", "mirror", "shampoo", "soap", "detergent", "laundry basket", "dryer", "hanger",
    
#     # Clothing
#     "shirt", "t-shirt", "pants", "jeans", "sweater", "jacket", "coat", "dress", "skirt", "shorts", "socks", "shoes", 
#     "boots", "gloves", "hat", "scarf", "tie", "belt", "sunglasses", "bag", "purse", "backpack", "wallet", "umbrella",
    
#     # Animals
#     "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "chicken", "duck", "elephant", "lion", "tiger", "bear", 
#     "rabbit", "hamster", "turtle", "lizard", "snake", "frog", "goat", "donkey", "kangaroo", "penguin", "zebra", "giraffe", 
#     "whale", "shark", "octopus", "jellyfish", "butterfly", "bee", "ant", "spider", "insect", "worm", "mouse", "rat", "bat",

#     # Food-related items
#     "apple", "banana", "orange", "grape", "lemon", "watermelon", "pineapple", "peach", "pear", "cherry", "strawberry", 
#     "blueberry", "raspberry", "tomato", "carrot", "lettuce", "broccoli", "spinach", "potato", "onion", "garlic", "pepper", 
#     "cucumber", "cabbage", "eggplant", "beetroot", "corn", "peas", "beans", "pumpkin", "mushroom", "rice", "bread", 
#     "pizza", "hamburger", "hotdog", "sandwich", "salad", "soup", "steak", "chicken", "pasta", "noodles", "burger", "cake", 
#     "cookie", "ice cream", "candy", "chocolate", "pie", "coffee", "tea", "juice", "water", "milk", "beer", "wine", "whiskey",

#     # Nature-related
#     "tree", "flower", "plant", "grass", "bush", "leaf", "wood", "rock", "mountain", "river", "lake", "ocean", "sky", 
#     "cloud", "rain", "snow", "sun", "moon", "star", "earth", "desert", "forest", "beach", "sand", "waterfall",

#     # Transportation
#     "car", "bus", "train", "airplane", "helicopter", "boat", "ship", "bicycle", "motorcycle", "truck", "van", "scooter", 
#     "subway", "tram", "taxi", "ambulance", "firetruck", "police car", "bicycle", "wheelchair", "skateboard", "rollerblades",

#     # Body parts
#     "eye", "nose", "ear", "tongue", "mouth", "hand", "finger", "arm", "leg", "foot", "toes", "neck", "shoulder", 
#     "back", "stomach", "chest", "knee", "ankle", "wrist", "elbow", "head", "hair", "skin", "teeth", "lip", "cheek", 
#     "chin", "forehead", "scalp", "throat", "bone", "muscle", "heart", "lungs", "liver", "kidney", "pancreas", "brain", 
#     "spine", "artery", "vein", "blood", "nail", "palm",

#     # Furniture & Electronics
#     "television", "computer", "radio", "refrigerator", "fan", "air conditioner", "microwave", "toaster", "coffee maker",
    
#     # Household appliances and tools
#     "vacuum cleaner", "iron", "sewing machine", "blender", "mixer", "grinder", "fan", "stove", "water heater", "kettle"
# ]

# model.set_classes(classes)

# # Text-to-Speech (TTS) Queue Setup
# tts_queue = queue.Queue()

# def tts_worker():
#     """Threaded function to process text-to-speech (TTS) queue."""
#     engine = pyttsx3.init()
#     while True:
#         text = tts_queue.get()
#         if text is None:
#             break  # Exit when None is received
#         engine.say(text)
#         engine.runAndWait()
#         tts_queue.task_done()

# tts_thread = threading.Thread(target=tts_worker, daemon=True)
# tts_thread.start()

# # Cooldown settings (in seconds)
# COOLDOWN = 5
# last_spoken = {}

# # Initialize annotators
# box_annotator = sv.BoxAnnotator(thickness=2)
# label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

# # Capture live webcam feed
# cap = cv2.VideoCapture(0)

# # Ensure valid webcam initialization
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break

#         # Resize frame for model compatibility
#         resized_frame = cv2.resize(frame, (640, 640))  # Ensure valid input size

#         # Run inference
#         try:
#             results = model.infer(resized_frame, confidence=0.25)
#             detections = sv.Detections.from_inference(results).with_nms(threshold=0.5)

#             if detections.class_id is None or len(detections.class_id) == 0:
#                 continue  # No objects detected

#         except Exception as e:
#             print(f"Error during inference: {e}")
#             continue

#         # Annotate frame
#         annotated_frame = frame.copy()
#         annotated_frame = box_annotator.annotate(annotated_frame, detections)

#         labels = []
#         for class_id, confidence in zip(detections.class_id, detections.confidence):
#             if 0 <= class_id < len(classes):
#                 label = f"{classes[class_id]} {confidence:.2f}"
#                 labels.append(label)
        
#         annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)

#         # Display the frame
#         cv2.imshow('Live Object Detection', annotated_frame)

#         # Provide audio output
#         current_time = time.time()
#         for class_id, confidence in zip(detections.class_id, detections.confidence):
#             if 0 <= class_id < len(classes):
#                 class_name = classes[class_id]
#                 if class_name not in last_spoken or (current_time - last_spoken[class_name] > COOLDOWN):
#                     tts_queue.put(f"Detected {class_name} with confidence {confidence:.2f}")
#                     last_spoken[class_name] = current_time

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#     tts_queue.put(None)  # Stop TTS thread
#     tts_thread.join()
import cv2
import supervision as sv
import queue
import threading
import pyttsx3
import time
import numpy as np
from inference.models.yolo_world.yolo_world import YOLOWorld

# Initialize YOLO World model with larger variant and optimized classes
model = YOLOWorld(model_id="yolo_world/l")  # Switched to larger model for better accuracy
optimized_classes = [
    # Consolidated categories
    "person", "child", "animal", 
    "chair", "table", "electronic device",
    "vehicle", "clothing", "food",
    "kitchen appliance", "bathroom item",
    "bag", "shoe", "plant"
 "person", "man", "woman", "child", "baby", "adult", "teenager", "elderly", "doctor", "nurse", "teacher", "student",
    
     # Household items
     "backpack", "chair", "table", "sofa", "bed", "tv", "remote", "microwave", "fridge", "oven", "fan", "toaster", 
     "dishwasher", "light", "lamp", "clock", "calendar", "bookshelf", "keyboard", "mouse", "computer", "laptop", 
     "phone", "tablet", "speaker", "camera", "headphones", "watch", "wallet", "glasses", "trashcan", "recycle bin", 
     "scissors", "plate", "fork", "knife", "spoon", "cup", "mug", "bottle", "glass", "napkin", "towel", "toothbrush", 
     "toothpaste", "comb", "razor", "mirror", "shampoo", "soap", "detergent", "laundry basket", "dryer", "hanger",
    
#     # Clothing
     "shirt", "t-shirt", "pants", "jeans", "sweater", "jacket", "coat", "dress", "skirt", "shorts", "socks", "shoes", 
     "boots", "gloves", "hat", "scarf", "tie", "belt", "sunglasses", "bag", "purse", "backpack", "wallet", "umbrella",
    
#     # Animals
     "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "chicken", "duck", "elephant", "lion", "tiger", "bear", 
     "rabbit", "hamster", "turtle", "lizard", "snake", "frog", "goat", "donkey", "kangaroo", "penguin", "zebra", "giraffe", 
     "whale", "shark", "octopus", "jellyfish", "butterfly", "bee", "ant", "spider", "insect", "worm", "mouse", "rat", "bat",

#     # Food-related items
     "apple", "banana", "orange", "grape", "lemon", "watermelon", "pineapple", "peach", "pear", "cherry", "strawberry", 
     "blueberry", "raspberry", "tomato", "carrot", "lettuce", "broccoli", "spinach", "potato", "onion", "garlic", "pepper", 
     "cucumber", "cabbage", "eggplant", "beetroot", "corn", "peas", "beans", "pumpkin", "mushroom", "rice", "bread", 
     "pizza", "hamburger", "hotdog", "sandwich", "salad", "soup", "steak", "chicken", "pasta", "noodles", "burger", "cake", 
     "cookie", "ice cream", "candy", "chocolate", "pie", "coffee", "tea", "juice", "water", "milk", "beer", "wine", "whiskey",

     # Nature-related
     "tree", "flower", "plant", "grass", "bush", "leaf", "wood", "rock", "mountain", "river", "lake", "ocean", "sky", 
     "cloud", "rain", "snow", "sun", "moon", "star", "earth", "desert", "forest", "beach", "sand", "waterfall",
     # Transportation
     "car", "bus", "train", "airplane", "helicopter", "boat", "ship", "bicycle", "motorcycle", "truck", "van", "scooter", 
     "subway", "tram", "taxi", "ambulance", "firetruck", "police car", "bicycle", "wheelchair", "skateboard", "rollerblades",

     # Body parts
     "eye", "nose", "ear", "tongue", "mouth", "hand", "finger", "arm", "leg", "foot", "toes", "neck", "shoulder", 
     "back", "stomach", "chest", "knee", "ankle", "wrist", "elbow", "head", "hair", "skin", "teeth", "lip", "cheek", 
     "chin", "forehead", "scalp", "throat", "bone", "muscle", "heart", "lungs", "liver", "kidney", "pancreas", "brain", 
     "spine", "artery", "vein", "blood", "nail", "palm",

     # Furniture & Electronics
     "television", "computer", "radio", "refrigerator", "fan", "air conditioner", "microwave", "toaster", "coffee maker",
    
     # Household appliances and tools
     "vacuum cleaner", "iron", "sewing machine", "blender", "mixer", "grinder", "fan", "stove", "water heater", "kettle"
 ]

model.set_classes(optimized_classes)

# Text-to-Speech (TTS) Queue Setup
tts_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Slower speech for clarity
    while True:
        text = tts_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Cooldown settings
COOLDOWN = 3  # Reduced cooldown for quicker updates
CONFIDENCE_THRESHOLD = 0.4  # Increased confidence threshold
last_spoken = {}

# Initialize annotators with better visibility
box_annotator = sv.BoxAnnotator(
    thickness=2,
    color=sv.Color(r=50, g=50, b=255)
)
label_annotator = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.8,
    text_color=sv.Color.WHITE,
    text_padding=5
)

# Webcam setup with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def letterbox_image(img, target_size=640):
    # Maintains aspect ratio with padding
    h, w = img.shape[:2]
    scale = min(target_size/h, target_size/w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
    
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Preprocess with letterboxing
        processed_frame = letterbox_image(frame)
        
        # Run inference with higher confidence threshold
        results = model.infer(
            processed_frame,
            confidence=CONFIDENCE_THRESHOLD,
            nms=0.4  # Stricter NMS threshold
        )
        detections = sv.Detections.from_inference(results).with_nms(0.4)

        # Filter and process detections
        current_time = time.time()
        filtered_detections = []
        labels = []
        
        for idx, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            if 0 <= class_id < len(optimized_classes):
                class_name = optimized_classes[class_id]
                
                # Confidence-based cooldown
                cooldown = COOLDOWN * (1 - confidence)
                if class_name in last_spoken and (current_time - last_spoken[class_name]) < cooldown:
                    continue
                
                last_spoken[class_name] = current_time
                filtered_detections.append(idx)
                labels.append(f"{class_name} {confidence:.2f}")
                tts_queue.put(f"{class_name} detected")

        # Apply filtering
        detections = detections[filtered_detections]

        # Annotate frame
        annotated_frame = box_annotator.annotate(
            frame.copy(), detections
        )
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections, labels=labels
        )

        # Display
        cv2.imshow('Enhanced Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    tts_thread.join()
    