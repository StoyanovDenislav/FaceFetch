import RPi.GPIO as GPIO
import time

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins
ENA = 5
IN1 = 17
IN2 = 18

ENB = 6
IN3 = 22
IN4 = 27

# Ultrasonic pins
TRIG = 23
ECHO = 24

GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Enable motors
GPIO.output(ENA, GPIO.HIGH)
GPIO.output(ENB, GPIO.HIGH)

# ---------------- MOTOR FUNCTIONS ----------------
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
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)

# ---------------- DISTANCE FUNCTION ----------------
def distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.04

    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return 999

    start = time.time()

    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return 999

    end = time.time()

    duration = end - start
    return (duration * 34300) / 2

# ---------------- MAIN LOOP ----------------
try:
    while True:
        dist = distance()
        print(f"Distance: {dist:.1f} cm")

        if dist > 20:
            forward()
        else:
            stop()
            time.sleep(0.3)
            turn_left()
            time.sleep(0.5)

except KeyboardInterrupt:
    stop()
    GPIO.cleanup()
