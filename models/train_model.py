import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -----------------------------------
# 7 Emotion Labels (Original FER)
# -----------------------------------
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

DATASET_PATH = "dataset/train"

def load_data(path):
    images, labels = [], []

    for emotion in emotion_labels:
        emotion_folder = os.path.join(path, emotion)

        if not os.path.exists(emotion_folder):
            continue

        label = emotion_labels.index(emotion)

        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (48, 48))
            img = img / 255.0

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# -----------------------------------
# Load Data
# -----------------------------------
X, y = load_data(DATASET_PATH)

X = X.reshape(-1, 48, 48, 1)
y = to_categorical(y, num_classes=7)

# -----------------------------------
# Train / Validation Split
# -----------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# CNN Model (7 Output Classes)
# -----------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")   # 7 emotions
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# Train Model
# -----------------------------------
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# -----------------------------------
# Save Model
# -----------------------------------
model.save("emotion_7class_model.h5")
print("Model saved as emotion_7class_model.h5")