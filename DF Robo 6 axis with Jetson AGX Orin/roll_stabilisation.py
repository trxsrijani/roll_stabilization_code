# Flask-based web application that reads roll angle from a DFROBOT 6-axis IMU sensor via UART,
# stabilizes live video feed by counter-rotating frames in real time, and streams it through a browser interface.
# No cropping done

import threading
import time
from flask import Flask, Response, jsonify, make_response
import cv2
import logging
from collections import deque
import statistics
import serial
import struct
import math

# ========== DFRobotIMU Class ==========
class DFRobotIMU:
    START_BYTE = 0x55
    HEADER_ANGLE = 0x53
    PACKET_LENGTH = 11

    def __init__(self, port="/dev/ttyTHS0", baudrate=9600, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
        except serial.SerialException as e:
            raise Exception(f"Error opening serial port {port}: {e}")
        time.sleep(0.5)
        self.buffer = bytearray()

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception as e:
            print(f"Error closing IMU serial port: {e}")

    def _calculate_checksum(self, packet):
        return sum(packet[0:10]) & 0xFF

    def _parse_angle_packet(self, packet):
        if len(packet) != self.PACKET_LENGTH:
            return None
        if packet[0] != self.START_BYTE or packet[1] != self.HEADER_ANGLE:
            return None
        if self._calculate_checksum(packet) != packet[10]:
            return None
        try:
            angle_x_raw = struct.unpack('<h', bytes(packet[2:4]))[0]
            angle_y_raw = struct.unpack('<h', bytes(packet[4:6]))[0]
            angle_z_raw = struct.unpack('<h', bytes(packet[6:8]))[0]
            angle_x = angle_x_raw / 32768.0 * 180.0
            angle_y = angle_y_raw / 32768.0 * 180.0
            angle_z = angle_z_raw / 32768.0 * 180.0
        except Exception as e:
            print(f"Error parsing angle values: {e}")
            return None
        return (angle_x, angle_y, angle_z)

    def read_angle(self, read_timeout=0.1, retries=3):
        for attempt in range(1, retries + 1):
            latest_angles = None
            start_time = time.time()

            while time.time() - start_time < read_timeout:
                try:
                    data = self.ser.read(self.ser.in_waiting or 1)
                except Exception as e:
                    print(f"IMU serial read error: {e}")
                    continue

                if data:
                    self.buffer.extend(data)

                while len(self.buffer) >= self.PACKET_LENGTH:
                    if self.buffer[0] != self.START_BYTE:
                        self.buffer.pop(0)
                        continue

                    packet = self.buffer[:self.PACKET_LENGTH]
                    del self.buffer[:self.PACKET_LENGTH]

                    try:
                        if packet[1] == self.HEADER_ANGLE:
                            angles = self._parse_angle_packet(packet)
                            if angles is not None:
                                latest_angles = angles
                    except Exception as e:
                        print(f"IMU packet parsing error: {e}")

                time.sleep(0.01)

            if latest_angles is not None:
                return latest_angles
        return (0.0, 0.0, 0.0)


# ========== Flask App ==========
tt = time.time
t0 = tt()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)

# Thread-safe angle storage
angle_lock = threading.Lock()
current_roll = 0.0


def sensor_loop():
    global current_roll
    imu = DFRobotIMU()
    while True:
        angle_x, angle_y, angle_z = imu.read_angle()
        roll = angle_x  # Assuming roll = angle_x
        if roll < 0:
            roll += 360
        with angle_lock:
            current_roll = roll
        time.sleep(0.02)


def generate_frames():
    global current_roll
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Camera not accessible")
        return

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

        normalized = (roll-180) % 360 - 180
        raw = -normalized

        history.append(raw)
        med = statistics.median(history)

        alpha_s = min(1.0, dt_f * 8.0)
        display_ang = prev_display + (med - prev_display) * alpha_s
        prev_display = display_ang

        (h, w) = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), display_ang, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        status = f"Roll: {roll:.1f}°"
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
