# Import necessary modules
import os  # Operating system interface for file and directory management
import pickle  # Module for serializing and deserializing Python objects
import mediapipe as mp  # MediaPipe for hand detection and landmark processing
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # Matplotlib for plotting (not used in this script)

# Initialize MediaPipe's hand detection and drawing utilities
mp_hands = mp.solutions.hands  # Hands solution from MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for visualization
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles for landmarks

# Configure MediaPipe Hands for static image processing
# IMPORTANT: max_num_hands and model_complexity must match inference (app.py)
hands = mp_hands.Hands(
    static_image_mode=True,         # True is correct here — we're processing saved images
    max_num_hands=1,                 # Must match inference: only process 1 hand
    model_complexity=1,              # Must match inference: full model
    min_detection_confidence=0.3,    # Low threshold to capture all training samples
)

# Define the directory where the dataset is stored
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []  # List to store processed landmark data
labels = []  # List to store corresponding labels for each data entry

# Loop through each class directory in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Auxiliary list to store normalized landmark coordinates for the current image
        x_ = []  # List to store x-coordinates of landmarks
        y_ = []  # List to store y-coordinates of landmarks

        # Read the image using OpenCV and convert it to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            # Only use the first (primary) hand — must match inference code in app.py
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract x and y coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize landmark coordinates relative to the hand's bounding box
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Only add this sample if we got exactly 42 features (21 landmarks × 2)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

# Serialize and save the processed data and labels using pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
