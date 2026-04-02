import os
# Suppress TensorFlow GPU and optimization warnings before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import cv2
import logging
import time
import threading
import imutils
import random
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Required for non-interactive plotting in threads
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import winsound for Windows alerts
try:
    import winsound
except ImportError:
    winsound = None

app = Flask(__name__, template_folder='.')

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'sample.keras')

# Global State & Locks
lock = threading.Lock()
data_lock = threading.Lock()
is_camera_on = True
detection_enabled = True
model = None
input_size = 224
has_model = False

def load_system_model():
    global model, input_size, has_model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found at {MODEL_PATH}. Detection disabled.")
        return False
    try:
        model = load_model(MODEL_PATH, compile=False)
        input_size = model.input_shape[1]
        has_model = True
        logging.info(f"Model loaded successfully. Input size: {input_size}")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return False

def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    return cap

# Initialization
load_system_model()
camera = get_camera()

# Analytics State
severity_data = deque(maxlen=1000)  # Keep only the last 1000 records to prevent memory issues
heatmap_coords = [] # List of (x, y) tuples

def get_mock_gps():
    """Simulates GPS coordinates for log entries."""
    lat = 12.9716 + random.uniform(-0.0005, 0.0005)
    lon = 77.5946 + random.uniform(-0.0005, 0.0005)
    return round(lat, 6), round(lon, 6)

def predict_potholes_batch(rois, model_obj, size):
    if not rois or model_obj is None: return []
    # Stack all regions into a single batch to process them in parallel
    batch = np.array([cv2.resize(r, (size, size)) for r in rois]).astype('float32') / 255.0
    predictions = model_obj.predict(batch, verbose=0)
    return [(np.argmax(p), np.max(p)) for p in predictions]

def play_alert_sound():
    """Triggers a beep on Windows systems."""
    if winsound:
        try:
            winsound.Beep(1000, 500)
        except Exception:
            pass

def generate_frames():
    global camera
    roi_h = 250
    top = 180
    bottom = top + roi_h
    # Reverted to original working ROI positions
    roi_x_positions = [50, 225, 400]
    last_alert_time = 0
    last_results = [(0, 0.0)] * len(roi_x_positions)
    frame_count = 0

    while True:
        if not is_camera_on:
            time.sleep(0.1)
            continue
            
        with lock:
            if camera is None or not camera.isOpened():
                time.sleep(0.1)
                continue
            success, frame = camera.read()

        if not success:
            time.sleep(0.1)
            continue
        
        # Resize for performance and flip for natural mirror view
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        if detection_enabled and has_model:
            if frame_count % 2 == 0:
                rois = [frame[top:bottom, x:x+roi_h] for x in roi_x_positions]
                last_results = predict_potholes_batch(rois, model, input_size)
            
            for i, (p_class, p_prob) in enumerate(last_results):
                x_start = roi_x_positions[i]
                try:
                    if p_class == 1 and p_prob > 0.75:
                        now = time.time()
                        if now - last_alert_time > 5.0:
                            threading.Thread(target=play_alert_sound, daemon=True).start()
                            last_alert_time = now

                        lat, lon = get_mock_gps()
                        with data_lock:
                            severity_data.append({
                                'timestamp': time.strftime("%H:%M:%S"),
                                'confidence': float(p_prob),
                                'lat': lat,
                                'lon': lon,
                                'x_pos': x_start
                            })
                            heatmap_coords.append((x_start + roi_h//2, top + roi_h//2))

                        cv2.rectangle(display_frame, (x_start, top), (x_start+roi_h, bottom), (0, 0, 255), 3)
                        cv2.putText(display_frame, f"POTHOLE: {p_prob*100:.1f}%", (x_start, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(display_frame, (x_start, top), (x_start+roi_h, bottom), (0, 255, 0), 1)
                except Exception as e:
                    logging.debug(f"Error processing ROI {i}: {e}")
            
            frame_count += 1

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera')
def toggle_camera():
    global is_camera_on, camera
    with lock:
        is_camera_on = not is_camera_on
        if not is_camera_on and camera:
            camera.release()
            camera = None
        elif is_camera_on and camera is None:
            camera = get_camera()
    return jsonify({"status": is_camera_on})

@app.route('/toggle_detection')
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({"status": detection_enabled})

@app.route('/log_data')
def log_data():
    with data_lock:
        if not severity_data:
            return jsonify([])
        
        df = pd.DataFrame(severity_data).tail(20) # Latest 20 records
    
    if len(df) > 5:
        try:
            iso = IsolationForest(contamination=0.1, random_state=42)
            df['anomaly'] = iso.fit_predict(df[['confidence']])
            df['status'] = df['anomaly'].apply(lambda x: 'ANOMALY' if x == -1 else 'STABLE')
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            df['status'] = 'ERROR'
    else:
        df['status'] = 'COLLECTING'
        
    return jsonify(df.to_dict(orient='records'))

@app.route('/heatmap')
def heatmap():
    with data_lock:
        coords = list(heatmap_coords)

    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    if coords:
        x, y = zip(*coords)
        hb = ax.hexbin(x, y, gridsize=15, cmap='magma', mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Frequency', color='white')
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title("Spatial Detection Density (X-Y Plane)", color='white', pad=10)
    ax.axis('off')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='#0d1117', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{plot_url}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)