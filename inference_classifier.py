# Import necessary modules
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model from pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Hands in tracking mode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        min_x, min_y = min(x_), min(y_)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

        # Scale-normalization (distance and hand-size invariant)
        data_aux = np.asarray(data_aux, dtype=np.float32)
        hand_scale = max(float(data_aux[0::2].max()), float(data_aux[1::2].max()))
        if hand_scale > 1e-4:
            data_aux /= hand_scale

        x1 = max(0, int(min_x * W) - 20)
        y1 = max(0, int(min_y * H) - 20)
        x2 = min(W, int(max(x_) * W) + 20)
        y2 = min(H, int(max(y_) * H) + 20)

        try:
            if len(data_aux) == 42:
                prediction = model.predict([data_aux])
                proba = model.predict_proba([data_aux])[0]
                conf = max(proba)
                predicted_character = labels_dict[int(prediction[0])]

                display_text = f"{predicted_character} ({conf*100:.0f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, display_text, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            pass

    cv2.imshow('Sign2Text Live Detection', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
