# import time
# import keyboard  # Requires the 'keyboard' library

# # Mock GPIO for testing on non-Raspberry Pi platforms
# class MockGPIO:
#     BCM = 'BCM'
#     IN = 'IN'
#     PUD_UP = 'PUD_UP'
#     FALLING = 'FALLING'
    
#     def setmode(self, mode):
#         print(f"GPIO setmode({mode})")
    
#     def setup(self, pin, mode, pull_up_down=None):
#         print(f"GPIO setup(pin={pin}, mode={mode}, pull_up_down={pull_up_down})")
    
#     def add_event_detect(self, pin, edge, callback=None, bouncetime=None):
#         print(f"GPIO add_event_detect(pin={pin}, edge={edge}, callback={callback}, bouncetime={bouncetime})")
    
#     def cleanup(self):
#         print("GPIO cleanup()")

# # Use MockGPIO for testing on non-Raspberry Pi platforms
# GPIO = MockGPIO()

# BUTTON_PIN = 2  # GPIO pin for hazard button
# PRESS_INTERVAL = 2  # Time interval in seconds within which two presses should be detected
# PHONE_NUMBER = "+919623164561"  # Replace with the recipient's phone number

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# def send_sms():
#     print(f"Sending SMS to {PHONE_NUMBER}: Emergency! Hazard button pressed.")

# def button_pressed():
#     print("Button pressed, sending SMS...")
#     send_sms()

# # Simulate button press using keyboard input
# keyboard.add_hotkey('`', button_pressed)

# try:
#     while True:
#         time.sleep(0.1)  # Small delay to prevent high CPU usage
# finally:
#     GPIO.cleanup()
#     print("Cleaning up...")
import time
import keyboard  # Requires the 'keyboard' library
import requests
from requests.auth import HTTPBasicAuth

# Mock GPIO for testing on non-Raspberry Pi platforms
class MockGPIO:
    BCM = 'BCM'
    IN = 'IN'
    PUD_UP = 'PUD_UP'
    FALLING = 'FALLING'
    
    def setmode(self, mode):
        print(f"GPIO setmode({mode})")
    
    def setup(self, pin, mode, pull_up_down=None):
        print(f"GPIO setup(pin={pin}, mode={mode}, pull_up_down={pull_up_down})")
    
    def add_event_detect(self, pin, edge, callback=None, bouncetime=None):
        print(f"GPIO add_event_detect(pin={pin}, edge={edge}, callback={callback}, bouncetime={bouncetime})")
    
    def cleanup(self):
        print("GPIO cleanup()")

# Use MockGPIO for testing on non-Raspberry Pi platforms
GPIO = MockGPIO()

BUTTON_PIN = 2  # GPIO pin for hazard button
PRESS_INTERVAL = 2  # Time interval in seconds within which three presses should be detected
PHONE_NUMBER = "+919623164561"  # Replace with the recipient's phone number

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Twilio credentials
twilioSID = "AC5d77abf2d08827d9367f7719ce355ff5"
twilioAuth = "a0e0203aa350e2f85b90104e1d2a0a7b"
twilioNumber = "+18314005482"
recipientNumber = "+919623164561"
message = "Emergency! The panic button has been pressed. Immediate action required!"

press_count = 0
first_press_time = 0

def send_sms():
    url = f"https://api.twilio.com/2010-04-01/Accounts/{twilioSID}/Messages.json"
    payload = {
        "To": recipientNumber,
        "From": twilioNumber,
        "Body": message
    }
    response = requests.post(url, data=payload, auth=HTTPBasicAuth(twilioSID, twilioAuth))
    print(f"HTTP Response Code: {response.status_code}")
    print(f"Server Response: {response.text}")

def button_pressed():
    global press_count, first_press_time
    current_time = time.time()
    
    if press_count == 0:
        first_press_time = current_time
        press_count += 1
        print("First press detected")
    elif press_count < 2 and (current_time - first_press_time) <= PRESS_INTERVAL:
        press_count += 1
        print(f"Press {press_count} detected")
    elif press_count == 2 and (current_time - first_press_time) <= PRESS_INTERVAL:
        print("Third press detected, sending SMS...")
        send_sms()
        press_count = 0  # Reset counter
    else:
        press_count = 1
        first_press_time = current_time
        print("First press detected")

# Simulate button press using keyboard input
keyboard.add_hotkey('1', button_pressed)

try:
    while True:
        time.sleep(0.1)  # Small delay to prevent high CPU usage
finally:
    GPIO.cleanup()
    print("Cleaning up...")