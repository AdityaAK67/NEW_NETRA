import lgpio
import time

TOUCH_SENSOR_PIN = 17  # GPIO17 (Pin 11)
GPIO_CHIP = 0  # Use gpiochip0

# Open GPIO chip
h = lgpio.gpiochip_open(GPIO_CHIP)

# Set pin as input (with pull-down resistor)
lgpio.gpio_claim_input(h, TOUCH_SENSOR_PIN)

print("Touch sensor test started. Press the sensor...")

try:
    while True:
        if lgpio.gpio_read(h, TOUCH_SENSOR_PIN) == 1:
            print("Touch detected!")
        time.sleep(0.1)  # Short delay to avoid excessive logging

except KeyboardInterrupt:
    print("\nTest stopped.")
    lgpio.gpiochip_close(h)  # Cleanup GPIO
