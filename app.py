import os
import time
import requests
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for, send_file, Response, session
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter
from datetime import datetime

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'

# Load YOLOv8 model
MODEL_PATH = r"C:\Users\SMART TECH\OneDrive - Higher Education Commission\Desktop\FDS\FabricDetectsion\runs\train\fabric_detect\weights\best.pt"
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    results = model.predict(source=image_path, conf=0.3, save=False, imgsz=416)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = []
    defect_types = []

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append((label, f"{conf:.2f}"))
                defect_types.append(label)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - 20), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(image, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    stats = {
        "total": len(defect_types),
        "by_type": dict(Counter(defect_types))
    }

    return image, detections, stats

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    prediction_url = None
    results_table = []
    stats = {"total": 0, "by_type": {}}
    detection_time = None

    if request.method == 'POST':
        image_path = None

        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                image_url = url_for('static', filename=f'uploads/{filename}')

        elif 'url' in request.form and request.form['url']:
            image_url_input = request.form['url']
            try:
                response = requests.get(image_url_input, stream=True, timeout=5)
                if response.status_code == 200:
                    filename = f"url_image_{int(time.time())}.jpg"
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    image_url = url_for('static', filename=f'uploads/{filename}')
            except:
                pass

        if image_path and os.path.exists(image_path):
            result_img, results_table, stats = predict_image(image_path)
            pred_filename = f"predicted_{int(time.time())}.jpg"
            prediction_path = os.path.join(app.config['UPLOAD_FOLDER'], pred_filename)
            cv2.imwrite(prediction_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            prediction_url = url_for('static', filename=f'uploads/{pred_filename}')
            session['download_path'] = prediction_path

            # Set detection time
            now = datetime.now()
            detection_time = now.strftime("%Y-%m-%d %H:%M:%S")

    return render_template("index.html",
                           image=image_url,
                           prediction=prediction_url,
                           results=results_table,
                           stats=stats,
                           detection_time=detection_time)

@app.route('/download')
def download():
    filepath = session.get('download_path')
    if filepath and os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.3, save=False, imgsz=416)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    text = f"{label} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, use_reloader=False)
