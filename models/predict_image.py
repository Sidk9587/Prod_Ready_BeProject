import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mental_health_model.h5")

classes = ["Healthy", "Neutral", "Depressed"]

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img / 255.0
    img = img.reshape(1,48,48,1)

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

# Example test image
image_path = r"dataset\test\happy\PrivateTest_218533.jpg"
print("Mental Health Status:", predict_image(image_path))
