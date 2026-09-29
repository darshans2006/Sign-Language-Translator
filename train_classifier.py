# Sign Language Classifier Training Script
# Uses MediaPipe landmarks (42 features per hand) + Scale Invariance + Flip Augmentation + Tuned RandomForest

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data from the pickle file
print("Loading dataset from data.pickle...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

raw_data = data_dict['data']
labels = np.asarray(data_dict['labels'])

FEATURE_LENGTH = 42
data = []
valid_labels = []
skipped = 0

for i, sample in enumerate(raw_data):
    sample_arr = np.asarray(sample, dtype=np.float32)
    if len(sample_arr) == FEATURE_LENGTH:
        data.append(sample_arr)
        valid_labels.append(labels[i])
    elif len(sample_arr) > FEATURE_LENGTH:
        data.append(sample_arr[:FEATURE_LENGTH])
        valid_labels.append(labels[i])
    else:
        skipped += 1

print(f"Loaded {len(data)} valid samples, skipped {skipped} bad samples.")

data = np.array(data, dtype=np.float32)
valid_labels = np.asarray(valid_labels)

# 1. Scale Normalization: make features invariant to distance from camera / hand size
print("Applying scale normalization (distance invariance)...")
normalized_data = []
for sample in data:
    xs = sample[0::2]
    ys = sample[1::2]
    scale = max(xs.max(), ys.max())
    if scale < 1e-4:
        scale = 1.0
    normalized_data.append(sample / scale)

normalized_data = np.array(normalized_data, dtype=np.float32)

# 2. Horizontal Flip Augmentation: allows detection of both Left & Right hands and mirrored webcams
print("Applying horizontal flip augmentation (Left/Right hand & mirror webcam invariance)...")
flipped_data = normalized_data.copy()
for i, sample in enumerate(normalized_data):
    xs = sample[0::2]
    flipped_data[i, 0::2] = xs.max() - xs

# Combine original + flipped samples
data_augmented = np.vstack([normalized_data, flipped_data])
labels_augmented = np.concatenate([valid_labels, valid_labels])

print(f"Augmented dataset size: {len(data_augmented)} samples across {len(set(labels_augmented))} classes.")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data_augmented, labels_augmented, test_size=0.2, shuffle=True, stratify=labels_augmented, random_state=42
)

# Initialize a tuned RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=250,       # 250 trees for maximum stability and robustness
    max_depth=None,         # Fully developed trees
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,              # All CPU cores
    random_state=42,
)

# Train the model
print("Training RandomForest model on augmented dataset...")
model.fit(x_train, y_train)

# Evaluate predictions
y_predict = model.predict(x_test)
test_proba = model.predict_proba(x_test)
test_conf = np.mean(test_proba.max(axis=1))

score = accuracy_score(y_predict, y_test)
print(f"Test Accuracy: {score * 100:.2f}%")
print(f"Average Prediction Confidence: {test_conf * 100:.2f}%")

# Save model dictionary with metadata
with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'scale_normalized': True,
        'flip_augmented': True,
        'feature_length': 42
    }, f)

print("Optimized model saved successfully to model.p!")
