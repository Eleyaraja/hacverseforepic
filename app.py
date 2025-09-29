# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import CategoricalFocalCrossentropy

# Constants from Colab
IMG_SIZE = 456
CLASS_NAMES = ['Myopia', 'Central Serous Chorioretinopathy-Color Fundus', 'Glaucoma', 'Disc Edema', 'Diabetic Retinopathy', 'Retinitis Pigmentosa', 'Macular Scar', 'Retinal Detachment', 'Healthy']
 # Update with your actual class names
model_path=r"C:\Users\Eleyaraja R\Downloads\best_model.h5"
# Custom layer definitions (must match training code)
class Cast(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    se = tf.keras.layers.Dense(channels // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    return tf.keras.layers.multiply([input_tensor, se])

def cbam_block(input_tensor, ratio=16):
    # Channel attention
    channel_avg = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    channel_max = tf.keras.layers.GlobalMaxPooling2D()(input_tensor)
    
    channel_avg = tf.keras.layers.Reshape((1, 1, -1))(channel_avg)
    channel_max = tf.keras.layers.Reshape((1, 1, -1))(channel_max)
    
    channel = tf.keras.layers.Concatenate()([channel_avg, channel_max])
    channel = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(channel)
    
    refined = tf.keras.layers.multiply([input_tensor, channel])
    
    # Spatial attention
    spatial_avg = tf.reduce_mean(refined, axis=-1, keepdims=True)
    spatial_max = tf.reduce_max(refined, axis=-1, keepdims=True)
    spatial = tf.keras.layers.Concatenate()([spatial_avg, spatial_max])
    spatial = tf.keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(spatial)
    
    return tf.keras.layers.multiply([refined, spatial])

# Image preprocessing functions
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = apply_clahe(image)
    image = image.astype(np.float32) / 255.0
    return image

# Load model with custom objects
@st.cache_resource
def load_model(model_path):
    custom_objects = {
        'Cast': Cast,
        'se_block': se_block,
        'cbam_block': cbam_block,
        'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Prediction function
def predict_image(model, image):
    processed = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed, axis=0))
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return CLASS_NAMES[predicted_class], confidence

# Streamlit UI
st.title("Retinal Disease Classification")
st.markdown("Upload a retinal image for disease classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load model
    model = load_model(model_path)  # Update with your model path
    
    # Read and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display image
    st.image(image, caption="Uploaded Image",  use_container_width=True)
    
    # Make prediction
    class_name, confidence = predict_image(model, image)
    
    # Show results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Disease", class_name)
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    # Show confidence distribution
    st.subheader("Class Confidence Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=CLASS_NAMES, y=model.predict(np.expand_dims(preprocess_image(image), axis=0))[0], ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

st.markdown("---")
st.info("Note: This model classifies retinal images into 15 categories of eye diseases. Always consult a medical professional for diagnosis.")