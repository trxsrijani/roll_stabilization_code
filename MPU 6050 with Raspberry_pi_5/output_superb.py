import threading
import time
from flask import Flask, Response, jsonify, make_response
import smbus
import math
import cv2
import logging
from collections import deque
import statistics

# Configure logging
tt = time.time
t0 = tt()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

# MPU6050 registers
MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_X = 0x3B
GYRO_X = 0x43
bus = smbus.SMBus(1)

# Fusion parameters
ALPHA = 0.98  # Complementary filter balance (accel vs gyro) (here 98% gyro, 2% accel used for roll)
prev_time = tt()
filtered_roll = 0.0

# Thread-safe angle storage
angle_lock = threading.Lock()
current_roll = 0.0


def mpu_init():
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
    # wake MPU6050 from sleep


def read_raw(addr):
    high = bus.read_byte_data(MPU_ADDR, addr)
    low = bus.read_byte_data(MPU_ADDR, addr + 1)
    val = (high << 8) | low
    return val - 65536 if val > 32767 else val


def sensor_loop():
    global filtered_roll, prev_time, current_roll
    mpu_init()
    while True:
        now = tt()
        dt = now - prev_time
        if dt <= 0:
            continue
        prev_time = now
        # Accelerometer
        ax = read_raw(ACCEL_X) / 16384.0
        ay = read_raw(ACCEL_X + 2) / 16384.0
        az = read_raw(ACCEL_X + 4) / 16384.0
        acc_roll = math.degrees(math.atan2(ay, az))
        if acc_roll<0:
            acc_roll+=360
        # print(f'Angle: {acc_roll}')

        # Gyroscope
        gx = read_raw(GYRO_X) / 131.0  # deg/s
        # Complementary filter fusion
        pred = filtered_roll + gx * dt
        filtered_roll = ALPHA * pred + (1 - ALPHA) * acc_roll
        with angle_lock:
            current_roll = filtered_roll
        time.sleep(0.005)


def generate_frames():
    global current_roll
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Camera not accessible")
        return

    # History for median filtering around horizon
    history = deque(maxlen=5)
    prev_display = 0.0
    prev_time_f = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now_f = time.time()
        dt_f = now_f - prev_time_f
        prev_time_f = now_f

        with angle_lock:
            roll = current_roll

        # map roll to raw angle (-90..+90 effectively)
        normalized = (roll)%360-180        # median filter to remove spikes
        raw = -normalized
        
        history.append(raw)
        med = statistics.median(history)

        # dynamic smoothing: faster when lag large, tighter when small
        alpha_s = min(1.0, dt_f * 8.0)
        display_ang = prev_display + (med - prev_display) * alpha_s
        prev_display = display_ang

        # apply rotation
        (h, w) = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), display_ang, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        # overlay status
        status = f"Roll: {roll:.1f}\u00b0"
        cv2.putText(rotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        ret2, buf = cv2.imencode('.jpg', rotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()

# Start sensor thread
t = threading.Thread(target=sensor_loop, daemon=True)
t.start()

@app.route('/')
def index():
    html = '''
    <!doctype html>
    <html><head><title>Upright View</title></head>
    <body style="margin:0; background:#000;
                 display:flex; justify-content:center;
                 align-items:center; height:100vh;">
      <img src="/video_feed" style="max-width:100%; height:auto;" />
    </body></html>
    '''
    return make_response(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/angle')
def get_angle():
    with angle_lock:
        r = round(current_roll, 2)
    return jsonify({'roll': r})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)