"""
Retrain Image Model — 7-Class Facial Emotion Recognition
=========================================================
Run this from the mental_health_fer/ folder:
    python models/retrain_image_model.py

This will:
  1. Load images from models/dataset/train/
  2. Build a deeper CNN
  3. Train for 30 epochs with data augmentation
  4. Evaluate on models/dataset/test/
  5. Save emotion_7class_model.h5 to models/

REQUIREMENT: pip install tensorflow opencv-python numpy
NOTE: Training takes ~30-60 min on CPU, ~5-10 min with GPU.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ── Label order MUST match app.py emotion_labels ─────────────────────────────
# app.py uses: ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
# Dataset folders: angry, disgust, fear, happy, neutral, sad, surprise
# We fix the order to exactly match app.py:
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
#                   0        1          2       3        4       5           6

DATASET_TRAIN = os.path.join("models", "dataset", "train")
DATASET_TEST  = os.path.join("models", "dataset", "test")
IMG_SIZE = 48

def load_split(path, labels):
    images, targets = [], []
    print(f"  Loading from: {path}")
    for label in labels:
        folder = os.path.join(path, label)
        if not os.path.exists(folder):
            print(f"    WARNING: folder not found: {folder}")
            continue
        files = os.listdir(folder)
        print(f"    {label}: {len(files)} images")
        for fname in files:
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img)        # histogram equalisation
            images.append(img)
            targets.append(labels.index(label))
    return np.array(images), np.array(targets)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[1/5] Loading training data...")
X_all, y_all = load_split(DATASET_TRAIN, EMOTION_LABELS)
print(f"      Total train images: {len(X_all)}")

print("[1/5] Loading test data...")
X_test, y_test = load_split(DATASET_TEST, EMOTION_LABELS)
print(f"      Total test images:  {len(X_test)}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
print("\n[2/5] Preprocessing...")
X_all  = X_all.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

y_all_cat  = to_categorical(y_all,  num_classes=7)
y_test_cat = to_categorical(y_test, num_classes=7)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all_cat, test_size=0.15, random_state=42, stratify=y_all
)
print(f"      Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ── 3. Data augmentation ──────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
datagen.fit(X_train)

# ── 4. Build model ────────────────────────────────────────────────────────────
print("\n[3/5] Building CNN model...")
model = Sequential([
    # Block 1
    Conv2D(64, (3,3), activation="relu", padding="same", input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(7, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\n[4/5] Training...")
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating on test set...")
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"      Test Accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join("models", "emotion_7class_model.h5")
model.save(out_path)
print(f"\n✅ Image model saved to: {out_path}")
print(f"\nLabel order (index 0-6):")
for i, lbl in enumerate(EMOTION_LABELS):
    print(f"  {i}: {lbl}")
print("\nMake sure app.py emotion_labels matches this order!")
print('app.py should have: emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]')
