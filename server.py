import cv2
import time
import base64
from flask import Flask, render_template, jsonify, request
from ultralytics import YOLO
from deepface import DeepFace

app = Flask(__name__)

model = YOLO("yolov8n-face.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Стартовые координаты по умолчанию
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 400, 150, 880, 650

emotions_ru = {
    'angry': 'Zloi', 'disgust': 'Otbraschenie', 'fear': 'Strah',
    'happy': 'Radost', 'sad': 'Grust', 'surprise': 'Udivlenie', 'neutral': 'Spokoen'
}

frame_count = 0
last_age, last_emotion = "...", "..."
person_was_present = False

current_event = None
latest_frame_base64 = ""
event_timestamp = ""

@app.route('/')
def index():
    return render_template('index.html')

# Принимаем новые координаты динамической зоны от клиента
@app.route('/update_roi', methods=['POST'])
def update_roi():
    global ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
    data = request.get_json()
    try:
        # Серверные координаты строго привязаны к исходному HD-кадру 1280x720
        ROI_X1 = int(data['x1'])
        ROI_Y1 = int(data['y1'])
        ROI_X2 = int(data['x2'])
        ROI_Y2 = int(data['y2'])
        return jsonify({'status': 'success', 'roi': [ROI_X1, ROI_Y1, ROI_X2, ROI_Y2]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/check_event')
def check_event():
    global current_event, latest_frame_base64, event_timestamp
    if current_event:
        response = {
            'event': current_event,
            'image': latest_frame_base64,
            'time': event_timestamp
        }
        current_event = None
        return jsonify(response)
    return jsonify({'event': None})

def generate_frames():
    global frame_count, last_age, last_emotion, person_was_present
    global current_event, latest_frame_base64, event_timestamp
    global ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        h, w, _ = frame.shape

        # Отрисовка текущей динамической зоны контроля
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
        cv2.putText(frame, "ZONA KONTROLYA", (ROI_X1 + 10, ROI_Y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Безопасная валидация кропа, чтобы избежать падения при пустых или некорректных размерах
        x1_crop = min(max(0, ROI_X1), w)
        y1_crop = min(max(0, ROI_Y1), h)
        x2_crop = min(max(0, ROI_X2), w)
        y2_crop = min(max(0, ROI_Y2), h)

        if x1_crop > x2_crop: x1_crop, x2_crop = x2_crop, x1_crop
        if y1_crop > y2_crop: y1_crop, y2_crop = y2_crop, y1_crop

        roi_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        person_currently_present = False

        if roi_crop.size > 0:
            results = model.predict(source=roi_crop, device='cpu', conf=0.3, imgsz=320, verbose=False)
            
            for result in results:
                if len(result.boxes) > 0:
                    person_currently_present = True
                    
                    for box in result.boxes:
                        coords = box.xyxy.tolist()
                        
                        if coords and len(coords) > 0:
                            # берем первый вложенный список
                            rx1, ry1, rx2, ry2 = map(int, coords[0])
                            
                            fx1 = x1_crop + rx1
                            fy1 = y1_crop + ry1
                            fx2 = x1_crop + rx2
                            fy2 = y1_crop + ry2

                            fx1, fy1 = max(0, fx1), max(0, fy1)
                            fx2, fy2 = min(w, fx2), min(h, fy2)

                            face_crop = frame[fy1:fy2, fx1:fx2]
                            if face_crop.size > 0 and frame_count % 5 == 0:
                                try:
                                    analysis_list = DeepFace.analyze(img_path=face_crop, actions=['age', 'emotion'], detector_backend='skip', enforce_detection=False, silent=True)
                                    analysis = analysis_list[0] if isinstance(analysis_list, list) else analysis_list
                                    last_age = str(int(analysis['age']))
                                    last_emotion = emotions_ru.get(analysis['dominant_emotion'], analysis['dominant_emotion'])
                                except:
                                    pass

                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Age: {last_age} | {last_emotion}", (fx1, fy1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if person_currently_present and not person_was_present:
            _, buffer_triggered = cv2.imencode('.jpg', frame)
            latest_frame_base64 = base64.b64encode(buffer_triggered).decode('utf-8')
            event_timestamp = time.strftime("%Y%m%d-%H%M%S")
            current_event = 'appeared'
            person_was_present = True
            
        elif not person_currently_present and person_was_present:
            _, buffer_triggered = cv2.imencode('.jpg', frame)
            latest_frame_base64 = base64.b64encode(buffer_triggered).decode('utf-8')
            event_timestamp = time.strftime("%Y%m%d-%H%M%S")
            current_event = 'disappeared'
            person_was_present = False

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return app.response_class(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
