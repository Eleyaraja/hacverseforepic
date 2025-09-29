import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Title
st.title("Retinal Disease Classifier (No TensorFlow Hub)")

# Load Keras model
@st.cache_resource
def load_retina_model():
    model = load_model("C:\Users\Eleyaraja R\Downloads\best_model.h5")  # Replace with your trained model path
    return model

model = load_retina_model()

# Upload image
uploaded_file = st.file_uploader("Upload a Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (224, 224))  # Use 299x299 if InceptionV3
    img_array = preprocess_input(resized.astype(np.float32))
    img_batch = np.expand_dims(img_array, axis=0)

    st.image(image, caption="Uploaded Image",  use_container_width=True)

    # Predict
    preds = model.predict(img_batch)
    predicted_class = np.argmax(preds, axis=1)[0]
    st.success(f"Predicted class: {predicted_class}")
