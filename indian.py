import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# ✅ Set Tesseract path (Update this if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ Detect coins using Hough Circle Transform
def detect_coins(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=15, maxRadius=120)

    coins = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            coin_roi = img[max(0, y-r):y+r, max(0, x-r):x+r]  # Prevent negative indices
            coins.append((x, y, r, coin_roi))
    return coins

# ✅ Detect notes based on color and aspect ratio
def detect_notes(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 1.5 < aspect_ratio < 2.3:
                note_roi = img[y:y+h, x:x+w]
                notes.append((x, y, w, h, note_roi))
    return notes

# ✅ Recognize currency denomination using OCR
def recognize_denomination(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    denominations = ['10', '20', '50', '100', '200', '500', '2000']
    
    for d in denominations:
        if d in text:
            return d
    return "Unknown"

# ✅ Display image using Matplotlib (Fallback for OpenCV GUI issues)
def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

# ✅ Initialize webcam for real-time detection
def detect_currency_realtime():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    print("🎥 Starting real-time currency detection... (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame.")
            break

        coins = detect_coins(frame)
        for x, y, r, roi in coins:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, 'Coin', (max(0, x-r), max(0, y-r)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        notes = detect_notes(frame)
        for x, y, w, h, roi in notes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            denomination = recognize_denomination(roi)
            cv2.putText(frame, f'₹{denomination}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        try:
            cv2.imshow('Live Currency Detection', frame)
        except cv2.error:
            show_image(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Program exited.")

# ✅ Run the program
if __name__ == "__main__":
    detect_currency_realtime()
