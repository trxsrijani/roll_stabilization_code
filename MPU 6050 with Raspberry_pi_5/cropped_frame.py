from flask import Flask, Response, jsonify, make_response, request
import smbus
import math
import cv2

app = Flask(__name__)

# MPU6050 setup
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
bus = smbus.SMBus(1)

JITTER_THRESHOLD = 3  # default, user-adjustable

# === Cropping Config ===
ORIG_WIDTH = 640
ORIG_HEIGHT = 480
CROP_SIZE = int(min(ORIG_WIDTH, ORIG_HEIGHT) / math.sqrt(2))  # 339

def mpu6050_init():
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_raw_data(addr):
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    if value > 32768:
        value -= 65536
    return value

def calculate_roll_pitch(ax, ay, az):
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
    return round(roll), round(pitch)

def get_tilt_angles():
    try:
        mpu6050_init()
        ax = read_raw_data(ACCEL_XOUT_H)   / 16384.0
        ay = read_raw_data(ACCEL_XOUT_H+2) / 16384.0
        az = read_raw_data(ACCEL_XOUT_H+4) / 16384.0
        return calculate_roll_pitch(ax, ay, az)
    except Exception:
        return "--", "--"

def crop_center(frame, size):
    h, w = frame.shape[:2]
    x1 = w//2 - size//2
    y1 = h//2 - size//2
    return frame[y1:y1+size, x1:x1+size]

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    last_angle = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        roll, pitch = get_tilt_angles()

        if isinstance(roll, (int, float)):
            if abs(roll) <= 45:
                desired_angle = -roll
                desired_status = f"Leveling ({roll}Â°)"
            else:
                sign = 1 if roll > 0 else -1
                desired_angle = sign * (180 - abs(roll))
                desired_status = f"Flipped ({roll}Â°)"
        else:
            desired_angle = 0
            desired_status = "No IMU"

        if abs(desired_angle - last_angle) > JITTER_THRESHOLD:
            angle = desired_angle
            status = desired_status
            last_angle = desired_angle
        else:
            angle = last_angle
            status = "Stable"

        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        # Crop to stable center square
        cropped = crop_center(rotated, CROP_SIZE)

        # Draw status text
        # cv2.putText(cropped, status, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buf = cv2.imencode('.jpg', cropped)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Live Feed: Upright View</title>
      <style>
        body { margin:0; padding:0; background:#111;
               display:flex; flex-direction:column; align-items:center; height:100vh; }
        #container { position:relative; }
        #video    { display:block; margin-top: 20px; }
        #controls {
          color: #fff; margin-top: 20px;
          font-family: monospace;
        }
      </style>
    </head>
    <body>
      <div id="container">
        <img id="video" src="/video_feed" width="339" height="339" />
      </div>
      <div id="controls">
        <label for="threshold">Jitter Threshold (Â°): </label>
        <input type="range" id="threshold" min="1" max="5" value="3" step="1">
        <span id="valueLabel">3Â°</span>
      </div>
      <script>
        const slider = document.getElementById('threshold');
        const valueLabel = document.getElementById('valueLabel');
        slider.oninput = function () {
          valueLabel.textContent = this.value + 'Â°';
          fetch('/set_threshold?value=' + this.value);
        };
      </script>
    </body>
    </html>
    '''
    return make_response(html_content)

@app.route('/set_threshold')
def set_threshold():
    global JITTER_THRESHOLD
    value = request.args.get('value', type=int)
    if value and 1 <= value <= 5:
        JITTER_THRESHOLD = value
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensors')
def get_sensor_data():
    roll, pitch = get_tilt_angles()
    return jsonify({"roll_angle": roll, "pitch_angle": pitch})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')