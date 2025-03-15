# import cv2
# import easyocr
# import pyttsx3

# # Initialize EasyOCR reader and text-to-speech engine
# reader = easyocr.Reader(['en'], gpu=False)
# engine = pyttsx3.init()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process frame for text detection
#     text_ = reader.readtext(frame)
#     detected_text = ""

#     for bbox, text, score in text_:
#         if score > 0.25:
#             # Draw bounding box
#             cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
#             cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
#             detected_text += text + " "

#     # Read detected text aloud
#     if detected_text:
#         print("Detected:", detected_text)  # Debugging
#         engine.say(detected_text)
#         engine.runAndWait()

#     # Show the live feed
#     cv2.imshow('Live Text Detection', frame)

#     # Press 'Q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Function to process the captured frame
def process_frame(frame):
    text_ = reader.readtext(frame)
    threshold = 0.25

    for bbox, text, score in text_:
        if score > threshold:
            cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Live Feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('z'):  # Capture frame and detect text when 'z' is pressed
        processed_frame = process_frame(frame.copy())
        cv2.imshow('Processed Frame', processed_frame)

    elif key == ord('q'):  # Quit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
