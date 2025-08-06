from flask import Flask, Response, jsonify, make_response
import smbus
import math
import cv2

app = Flask(__name__)

# MPU6050 setup
MPU6050_ADDR = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
bus = smbus.SMBus(1)

def mpu6050_init():
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_raw_data(addr):
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low  = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    if value > 32768:
        value -= 65536
    return value

def calculate_roll_pitch(ax, ay, az):
    # ax, ay, az already normalized (g units)
    # Roll  = atan2(ay, az)
    # Pitch = atan2(-ax, sqrt(ay^2 + az^2))
    roll  = math.degrees(math.atan2(ay, az))
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

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Camera not accessible")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        roll, pitch = get_tilt_angles()  # we only use roll here

        # decide rotation angle for continuous leveling + flip logic
        if isinstance(roll, (int, float)):
            if abs(roll) <= 90:
                angle = -roll
                status = f"Leveling ({roll}Â°)"
            else:
                sign = 1 if roll > 0 else -1
                angle = sign * (180 - abs(roll))
                status = f"Flipped ({roll}Â°)"
        else:
            angle = 0
            status = "No IMU"

        # apply rotation
        (h, w) = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # overlay status
        x1, y1 = w - 300, 40
        overlay = rotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x1+280, y1+80), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, rotated, 0.6, 0, rotated)
        cv2.putText(
            rotated, status, (x1+10, y1+50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2
        )

        ret, buf = cv2.imencode('.jpg', rotated)
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
        body { margin:0; padding:0; background:#111; display:flex;
               justify-content:center; align-items:center; height:100vh; }
        #container { position:relative; }
        #video    { display:block; }
        #overlay  { position:absolute; top:10px; right:10px;
                    background:rgba(0,0,0,0.4); color:#fff;
                    padding:10px 15px; font:16px/1 monospace;
                    border-radius:10px; }
      </style>
    </head>
    <body>
      <div id="container">
        <img id="video" src="/video_feed" />
        <div id="overlay">Frame auto-levels via IMU</div>
      </div>
    </body>
    </html>
    '''
    return make_response(html_content)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sensors')
def get_sensor_data():
    roll, pitch = get_tilt_angles()
    return jsonify({
        "roll_angle": roll,
        "pitch_angle": pitch
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')