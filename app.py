# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model and precompute class mappings
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    model_classes = [str(c) for c in model.classes_]
except Exception as e:
    print("Error loading the model:", e)
    model = None
    model_classes = []

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

# Pre-map model class indices to friendly text names
class_names = [labels_dict.get(int(c), str(c)) for c in model_classes] if model_classes else []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution for crisp landmark detection and high framerate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize MediaPipe Hands in tracking mode
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5,
    )

    # Motion detection & Temporal Smoothing variables
    prev_landmarks = None
    smoothed_coords = None
    proba_history = deque(maxlen=6)
    
    active_prediction = "..."
    active_confidence = 0.0
    pending_prediction = None
    pending_count = 0
    
    frame_count = 0
    last_emitted_pred = None

    MOTION_THRESHOLD = 0.022  # normalized movement threshold for motion detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror selfie view
        frame = cv2.flip(frame, 1)
        frame_count += 1

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw standard skeleton connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Raw current normalized coordinates (21, 2)
            current_coords = np.array(
                [[lm.x, lm.y] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )

            # 1. Motion Detection: measure displacement from previous frame
            if prev_landmarks is not None:
                displacement = np.linalg.norm(current_coords - prev_landmarks, axis=1)
                motion_speed = float(np.mean(displacement))
            else:
                motion_speed = 0.0
            prev_landmarks = current_coords.copy()

            is_moving = motion_speed > MOTION_THRESHOLD

            # 2. Exponential Moving Average (EMA) Landmark Smoothing
            # Smooths out camera noise/hand tremor while retaining quick motion response
            if smoothed_coords is None:
                smoothed_coords = current_coords.copy()
            else:
                alpha = 0.70  # 70% current frame, 30% history
                smoothed_coords = alpha * current_coords + (1.0 - alpha) * smoothed_coords

            # 3. Extract Scale-Invariant and Translation-Invariant Features
            xs = smoothed_coords[:, 0]
            ys = smoothed_coords[:, 1]
            min_x, min_y = float(np.min(xs)), float(np.min(ys))

            data_aux = []
            for i in range(21):
                data_aux.append(float(smoothed_coords[i, 0] - min_x))
                data_aux.append(float(smoothed_coords[i, 1] - min_y))

            data_aux = np.asarray(data_aux, dtype=np.float32)

            # Scale normalization: divide by maximum bounding dimension (distance/size invariant)
            hand_scale = max(float(data_aux[0::2].max()), float(data_aux[1::2].max()))
            if hand_scale > 1e-4:
                data_aux /= hand_scale

            # Bounding box for visual display
            x1 = max(0, int(min_x * W) - 20)
            y1 = max(0, int(min_y * H) - 20)
            x2 = min(W, int(np.max(xs) * W) + 20)
            y2 = min(H, int(np.max(ys) * H) + 20)

            # 4. Model Prediction & Motion-Aware Temporal Stabilization
            if model is not None and len(data_aux) == 42:
                try:
                    proba = model.predict_proba([data_aux])[0]
                    proba_history.append(proba)

                    # Compute smoothed average probability distribution over recent frames
                    mean_proba = np.mean(proba_history, axis=0)
                    best_idx = int(np.argmax(mean_proba))
                    best_conf = float(mean_proba[best_idx])
                    candidate_label = class_names[best_idx]

                    # Hysteresis confirmation: prevents jitter between similar signs
                    if candidate_label == active_prediction:
                        active_confidence = best_conf
                        pending_prediction = None
                        pending_count = 0
                    else:
                        if candidate_label == pending_prediction:
                            pending_count += 1
                        else:
                            pending_prediction = candidate_label
                            pending_count = 1

                        # Require 2 consecutive frames or very high confidence (>0.75) to switch
                        if pending_count >= 2 or best_conf > 0.75:
                            active_prediction = candidate_label
                            active_confidence = best_conf
                            pending_count = 0

                    # If hand is in rapid motion, keep the current prediction and mark status
                    status_text = "Moving" if is_moving else "Steady"

                    # 5. Visual UI Overlay on Camera Feed
                    box_color = (0, 230, 115) if not is_moving else (255, 175, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Top Banner
                    display_str = f"{active_prediction}  {active_confidence * 100:.0f}%  [{status_text}]"
                    (tw, th), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    banner_y1 = max(0, y1 - th - 14)
                    banner_y2 = y1
                    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 16, banner_y2), box_color, -1)
                    cv2.putText(
                        frame,
                        display_str,
                        (x1 + 8, banner_y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (15, 23, 42),  # dark slate text on bright banner
                        2,
                        cv2.LINE_AA
                    )

                    # 6. Throttle SocketIO emits (every 3 frames)
                    if frame_count % 3 == 0:
                        socketio.emit('prediction', {
                            'text': active_prediction,
                            'confidence': float(active_confidence),
                            'motion': status_text
                        })
                        last_emitted_pred = active_prediction

                except Exception as e:
                    pass

        else:
            # No hand in frame: reset tracker smoothly
            prev_landmarks = None
            smoothed_coords = None
            proba_history.clear()
            pending_count = 0

            if active_prediction != "...":
                active_prediction = "..."
                active_confidence = 0.0

            if frame_count % 10 == 0 and last_emitted_pred != "...":
                socketio.emit('prediction', {'text': '...', 'confidence': 0.0, 'motion': 'None'})
                last_emitted_pred = "..."

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
