import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("cnn8grps_rad1_model.h5")

def preprocess_image(image_path):
    """Preprocess the input image for model prediction."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (400, 400))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Test with a sample image
sample_image_path = "sample_inputs/test_image.jpg"  # Change as per available file
image = preprocess_image(sample_image_path)

# Predict
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

print(f"Predicted Class: {predicted_class}")
