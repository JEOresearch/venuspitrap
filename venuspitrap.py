from picamera2 import Picamera2
import cv2
import numpy as np
import pigpio
import time
import subprocess
import threading
import os

class ServoController(threading.Thread):
    def __init__(self, pi, gpio1, gpio2):
        super().__init__()
        self.pi = pi
        self.gpio1 = gpio1
        self.gpio2 = gpio2
        self._trigger = threading.Event()
        self._stop = threading.Event()
        self.cooldown_time = 4
        self.last_trigger = 0

    def run(self):
        while not self._stop.is_set():
            if self._trigger.wait(timeout=0.1):
                if time.time() - self.last_trigger >= self.cooldown_time:
                    self.last_trigger = time.time()
                    self.activate_servos()
                self._trigger.clear()

    def activate_servos(self):
        print("Servo thread: activating servos")
        self.pi.set_servo_pulsewidth(self.gpio1, 1250)
        self.pi.set_servo_pulsewidth(self.gpio2, 750)
        time.sleep(0.05)
        self.pi.set_servo_pulsewidth(self.gpio1, 0)
        self.pi.set_servo_pulsewidth(self.gpio2, 0)
        time.sleep(2)
        self.pi.set_servo_pulsewidth(self.gpio1, 1500)
        self.pi.set_servo_pulsewidth(self.gpio2, 500)
        time.sleep(0.05)
        self.pi.set_servo_pulsewidth(self.gpio1, 0)
        self.pi.set_servo_pulsewidth(self.gpio2, 0)

    def trigger(self):
        self._trigger.set()

    def stop(self):
        self._stop.set()

# Start pigpio daemon if not running
pi = pigpio.pi()
if not pi.connected:
    print("pigpiod not running. Starting now...")
    subprocess.run(["sudo", "pigpiod"])
    time.sleep(1)
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("Could not connect to pigpio daemon.")

# Open both cameras
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(2)
width, height = 640, 480
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cam1.isOpened() or not cam2.isOpened():
    print("One or both cameras could not be opened.")
    exit()

# Setup video writer settings
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
clip_length = 10  # seconds
clip_index = 0
clip_start_time = time.time()
detection_in_clip = False

# Create initial writer
def create_new_writer(index):
    return cv2.VideoWriter(f"camera1_clip_{index:03d}.avi", fourcc, fps, (width, height))

out = create_new_writer(clip_index)

# Setup detection
bbox_x, bbox_y = 245, 215
bbox_w, bbox_h = 120, 120
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, detectShadows=False)

# Servo setup
SERVO1_GPIO = 17
SERVO2_GPIO = 18
servo_thread = ServoController(pi, SERVO1_GPIO, SERVO2_GPIO)
servo_thread.start()

try:
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not (ret1 and ret2):
            print("Failed to read from cameras.")
            continue

        # Write frame1 and track recording time
        out.write(frame1)
        if time.time() - clip_start_time >= clip_length:
            out.release()
            if not detection_in_clip:
                os.remove(f"camera1_clip_{clip_index:03d}.avi")
                print(f"Deleted clip {clip_index:03d} due to no detection.")
            clip_index += 1
            out = create_new_writer(clip_index)
            clip_start_time = time.time()
            detection_in_clip = False

        # Detection logic (ROI-only processing)
        gray = cv2.cvtColor(frame2[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w], cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        white_pixel_count = np.sum(thresh == 255)

        valid_blob_detected = 50 <= white_pixel_count <= 200
        if valid_blob_detected:
            detection_in_clip = True
            servo_thread.trigger()

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cam1.release()
    cam2.release()
    out.release()
    servo_thread.stop()
    servo_thread.join()
    pi.stop()
    cv2.destroyAllWindows()
