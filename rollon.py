import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub  
from tensorflow_hub.keras_layer import KerasLayer  
from tensorflow.keras.losses import CategoricalFocalCrossentropy
import matplotlib.pyplot as plt
import seaborn as sns  


# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Custom objects needed for model loading
class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Dense(channels // reduction, activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.multiply([input_tensor, se])

def cbam_block(input_tensor, reduction_ratio=8):
    channel = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    channel = tf.keras.layers.Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu')(channel)
    channel = tf.keras.layers.Dense(input_tensor.shape[-1], activation='sigmoid')(channel)
    channel = tf.keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(channel)
    return tf.keras.layers.multiply([input_tensor, channel])

# Load the trained model

# Serialization decorators for custom components
@tf.keras.utils.register_keras_serializable()
class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

@tf.keras.utils.register_keras_serializable()
def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Dense(channels // reduction, activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.multiply([input_tensor, se])

@tf.keras.utils.register_keras_serializable()
def cbam_block(input_tensor, reduction_ratio=8):
    channel = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    channel = tf.keras.layers.Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu')(channel)
    channel = tf.keras.layers.Dense(input_tensor.shape[-1], activation='sigmoid')(channel)
    channel = tf.keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(channel)
    return tf.keras.layers.multiply([input_tensor, channel])

# Load the trained model with Hub support
@st.cache_resource
def load_model():
    custom_objects = {
        'Cast': Cast,
        'se_block': se_block,
        'cbam_block': cbam_block,
        'KerasLayer': KerasLayer,  # Added Hub layer
        'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy
    }
    
    model_path = r"C:\Users\Eleyaraja R\Downloads\best_model.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Rest of the code remains the same...
# Image preprocessing functions
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (456, 456))
    img = apply_clahe(img)
    img = img.astype(np.float32) / 255.0
    return img

# Streamlit UI
st.set_page_config(page_title="Retinal Image Classifier", layout="wide")
st.title("Retinal Disease Classification System")

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image",  use_container_width=True)
        
    # Preprocess and predict
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    if model is not None:
        prediction = model.predict(processed_image)
        class_names = ['Class0', 'Class1', 'Class2', 'Class3']  # Update with your actual class names
        confidence = np.max(prediction)
        predicted_class = class_names[np.argmax(prediction)]
        
        with col2:
            st.markdown(f"**Prediction:** {predicted_class}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            
            # Show prediction probabilities
            fig, ax = plt.subplots()
            ax.barh(class_names, prediction[0])
            ax.set_xlabel('Probability')
            ax.set_title('Class Probabilities')
            st.pyplot(fig)
    else:
        st.error("Model failed to load. Please check model file path.")

# Add dataset analysis section
if st.checkbox("Show Dataset Analysis"):
    dataset_path = r"C:\Users\Eleyaraja R\Desktop\docverse 3\Augmented Dataset"
    classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
    class_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in classes}
    
    st.subheader("Dataset Class Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
