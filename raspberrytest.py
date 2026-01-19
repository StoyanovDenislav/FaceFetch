import RPi.GPIO as GPIO
import time

# -----------------------------
# MOTOR PINS
# -----------------------------
# Left motor
IN1 = 17
IN2 = 18
ENA = 5

# Right motor
IN3 = 22
IN4 = 27
ENB = 6

# Ultrasonic sensor pins
TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup motor pins
GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT)

# Setup ultrasonic pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Stabilize TRIG
GPIO.output(TRIG, False)
time.sleep(0.1)

# Enable motors
GPIO.output(ENA, GPIO.HIGH)
GPIO.output(ENB, GPIO.HIGH)

# -----------------------------
# MOTOR CONTROL FUNCTIONS
# -----------------------------
def forward():
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)

def stop():
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)

def turn_left():
    # Pivot turn left
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)

# -----------------------------
# ULTRASONIC DISTANCE FUNCTION
# -----------------------------
def get_distance():
    # Send trigger pulse
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Timeout values
    timeout_start = time.time() + 0.02

    # Wait for ECHO HIGH
    while GPIO.input(ECHO) == 0:
        start = time.time()
        if start > timeout_start:
            return 999  # no echo

    timeout_end = time.time() + 0.02

    # Wait for ECHO LOW
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
        if stop_time > timeout_end:
            return 999  # no echo

    elapsed = stop_time - start
    distance = (elapsed * 34300) / 2  # cm
    return distance

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        dist = get_distance()
        print("Distance:", round(dist, 2), "cm")

        if dist < 20:
            print("Obstacle detected! Turning...")
            stop()
            time.sleep(0.3)

            turn_left()
            time.sleep(0.6)

            stop()
            time.sleep(0.2)
        else:
            forward()

        time.sleep(0.05)

except KeyboardInterrupt:
    stop()
    GPIO.cleanup()
