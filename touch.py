import RPi.GPIO as GPIO
import time

TOUCH_SENSOR_PIN = 17  # GPIO17 (Pin 11)

GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setup(TOUCH_SENSOR_PIN, GPIO.IN)  # Set as input

print("Touch sensor test started. Press the sensor...")

try:
    while True:
        if GPIO.input(TOUCH_SENSOR_PIN) == GPIO.HIGH:
            print("Touch detected!")
        time.sleep(0.1)  # Short delay to avoid excessive logging

except KeyboardInterrupt:
    print("\nTest stopped.")
    GPIO.cleanup()  # Reset GPIO settings
