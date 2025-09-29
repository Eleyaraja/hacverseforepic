import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

# Set these before TF imports
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Changed to 1 for compatibility
os.environ['TF_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Constants
IMG_SIZE = 456
MODEL_PATH = r"c:\Users\Eleyaraja R\Downloads\best_model.h5"
TRAIN_DATA_PATH = r"C:\Users\Eleyaraja R\Desktop\docverse 3\Augmented Dataset"

# Custom Cast Layer with serialization
@register_keras_serializable()
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self.dtype})
        return config

# Squeeze-and-Excitation Block with serialization
@register_keras_serializable()
class SEBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Reshape((1, 1, self.filters))
        self.dense1 = tf.keras.layers.Dense(self.filters // self.ratio, 
                                          activation='relu',
                                          kernel_initializer='he_normal')
        self.dense2 = tf.keras.layers.Dense(self.filters, 
                                          activation='sigmoid',
                                          kernel_initializer='he_normal')
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        se = self.gap(inputs)
        se = self.reshape(se)
        se = self.dense1(se)
        se = self.dense2(se)
        return self.multiply([inputs, se])

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config

# CBAM Block with serialization
@register_keras_serializable()
class CBAMBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        # Channel attention components
        self.channel_gap = tf.keras.layers.GlobalAveragePooling2D()
        self.channel_gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.channel_reshape = tf.keras.layers.Reshape((1, 1, input_shape[-1]))
        self.channel_dense1 = tf.keras.layers.Dense(input_shape[-1] // self.ratio, 
                                                  activation='relu')
        self.channel_dense2 = tf.keras.layers.Dense(input_shape[-1], 
                                                  activation='sigmoid')
        self.channel_add = tf.keras.layers.Add()
        self.channel_multiply = tf.keras.layers.Multiply()
        
        # Spatial attention components
        self.spatial_lambda_mean = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))
        self.spatial_lambda_max = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))
        self.spatial_concat = tf.keras.layers.Concatenate(axis=-1)
        self.spatial_conv = tf.keras.layers.Conv2D(1, kernel_size=7, 
                                                 padding='same', 
                                                 activation='sigmoid')
        self.spatial_multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        # Channel attention
        gap = self.channel_gap(inputs)
        gmp = self.channel_gmp(inputs)
        gap = self.channel_reshape(gap)
        gmp = self.channel_reshape(gmp)
        
        gap = self.channel_dense2(self.channel_dense1(gap))
        gmp = self.channel_dense2(self.channel_dense1(gmp))
        
        channel_attention = self.channel_add([gap, gmp])
        channel_refined = self.channel_multiply([inputs, channel_attention])
        
        # Spatial attention
        mean = self.spatial_lambda_mean(channel_refined)
        max_ = self.spatial_lambda_max(channel_refined)
        concat = self.spatial_concat([mean, max_])
        spatial_attention = self.spatial_conv(concat)
        
        return self.spatial_multiply([channel_refined, spatial_attention])

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config

# Load class names from folder structure
def get_class_names(train_path):
    try:
        classes = sorted([d.name for d in Path(train_path).iterdir() if d.is_dir()])
        if not classes:
            raise ValueError("No class folders found")
        return classes
    except Exception as e:
        st.error(f"Error loading classes: {str(e)}")
        return ['Myopia', 'Central Serous Chorioretinopathy-Color Fundus', 'Glaucoma', 
                'Disc Edema', 'Diabetic Retinopathy', 'Retinitis Pigmentosa', 
                'Macular Scar', 'Retinal Detachment', 'Healthy']

CLASS_NAMES = get_class_names(TRAIN_DATA_PATH)

# CLAHE preprocessing
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# Image preprocessing pipeline
def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = apply_clahe(image)
    image = image.astype(np.float32) / 255.0
    
    return image

# Model loading with all custom objects
@st.cache_resource
def load_model(model_path):
    custom_objects = {
        'Cast': Cast,
        'SEBlock': SEBlock,
        'CBAMBlock': CBAMBlock,
        'CategoricalFocalCrossentropy': tf.keras.losses.CategoricalFocalCrossentropy,
        'Adam': tf.keras.optimizers.Adam,
    }
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Verify input shape
        if model.input_shape[1:] != (IMG_SIZE, IMG_SIZE, 3):
            st.error(f"Model expects {model.input_shape[1:]}, but preprocessing outputs {(IMG_SIZE, IMG_SIZE, 3)}")
            return None
        
        # Recompile with same settings as original
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=9.999999747378752e-06),
            loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Prediction function
def predict_image(model, image):
    try:
        processed_img = preprocess_image(image)
        input_tensor = np.expand_dims(processed_img, axis=0)
        
        # Debug info
        st.session_state.processed_img = processed_img
        st.write("Preprocessed image stats:")
        st.write(f"Shape: {processed_img.shape} | Type: {processed_img.dtype}")
        st.write(f"Range: {processed_img.min():.4f}-{processed_img.max():.4f}")
        
        # Predict
        prediction = model.predict(input_tensor)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return CLASS_NAMES[predicted_class], confidence, prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# Main processing function
def process_retinal_image(image):
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    if st.checkbox("Show preprocessing details"):
        st.image(st.session_state.processed_img, 
                caption="Preprocessed Image", 
                clamp=True)
    
    class_name, confidence, probs = predict_image(model, image)
    
    if class_name:
        st.success(f"Prediction: {class_name} (Confidence: {confidence:.2%})")
        
        fig, ax = plt.subplots(figsize=(10,5))
        bars = ax.bar(CLASS_NAMES, probs, color='skyblue')
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        st.pyplot(fig)

# Streamlit UI
def main():
    st.title("👁️ Retinal Disease Detection")
    st.write(f"Detecting {len(CLASS_NAMES)} conditions: {', '.join(CLASS_NAMES)}")
    
    uploaded_file = st.file_uploader("Upload retinal fundus image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",  use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            process_retinal_image(image)

if __name__ == "__main__":
    main()