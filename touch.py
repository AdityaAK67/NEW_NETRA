# import RPi.GPIO as GPIO
# import time

# TOUCH_SENSOR_PIN = 17  # GPIO17 (Pin 11)

# GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
# GPIO.setup(TOUCH_SENSOR_PIN, GPIO.IN)  # Set as input

# print("Touch sensor test started. Press the sensor...")

# try:
#     while True:
#         if GPIO.input(TOUCH_SENSOR_PIN) == GPIO.HIGH:
#             print("Touch detected!")
#         time.sleep(0.1)  # Short delay to avoid excessive logging

# except KeyboardInterrupt:
#     print("\nTest stopped.")
#     GPIO.cleanup()  # Reset GPIO settings


import pigpio
import time

TOUCH_SENSOR_PIN = 17  # GPIO17 (Pin 11)
pi = pigpio.pi()

if not pi.connected:
    print("Error: pigpio daemon not running!")
    exit()

pi.set_mode(TOUCH_SENSOR_PIN, pigpio.INPUT)

print("Touch sensor test started. Press the sensor...")

try:
    while True:
        if pi.read(TOUCH_SENSOR_PIN):  # Read GPIO state
            print("Touch detected!")
        time.sleep(0.1)  

except KeyboardInterrupt:
    print("\nTest stopped.")
    pi.stop()  # Clean up pigpio connection
