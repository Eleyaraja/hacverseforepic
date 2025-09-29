import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
import nibabel as nib
import tempfile
import os
from pathlib import Path
import plotly.express as px
from scipy.ndimage import rotate
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Alzheimer's Disease Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def load_nifti_file(file_path):
    """Load NIfTI file and return the data array"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data.astype(np.float32)
    except Exception as e:
        st.error(f"Error loading NIfTI file: {str(e)}")
        return None

def preprocess_volume(volume):
    """Preprocess volume for model prediction"""
    try:
        # Apply 90-degree rotation (ADNI standard)
        volume = rotate(volume, 90, axes=(0, 1), reshape=False)
        
        # Normalize to [0, 1]
        volume_min, volume_max = np.min(volume), np.max(volume)
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Add batch and channel dimensions
        volume = np.expand_dims(volume, axis=0)  # Batch dimension
        volume = np.expand_dims(volume, axis=-1)  # Channel dimension
        
        return volume
    except Exception as e:
        st.error(f"Error preprocessing volume: {str(e)}")
        return None

def create_slice_visualization(volume, title="Brain Slices"):
    """Create a simple slice visualization"""
    try:
        if volume is None:
            return None
            
        # Remove extra dimensions
        if len(volume.shape) > 3:
            volume = np.squeeze(volume)
        
        # Select representative slices
        depth = volume.shape[2]
        slice_indices = np.linspace(10, depth-10, 12, dtype=int)
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        fig.suptitle(title, fontsize=16)
        
        for i, slice_idx in enumerate(slice_indices):
            row, col = i // 4, i % 4
            axes[row, col].imshow(volume[:, :, slice_idx], cmap='gray')
            axes[row, col].set_title(f'Slice {slice_idx}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<p class="big-font">🧠 Alzheimer\'s Disease 3D CNN Classifier</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Upload your trained model and brain scan to get AI-powered analysis**
    
    This application uses deep learning to classify brain scans for Alzheimer's disease detection.
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    # Left column - Model loading
    with col1:
        st.header("🤖 Model Loading")
        
        # Option 1: Load from local file
        st.subheader("Option 1: Load Local Model")
        model_path = st.text_input("Enter model path:", value=r"C:\Users\Eleyaraja R\Downloads\best.h5")
        
        if st.button("Load Model from Path"):
            if os.path.exists(model_path):
                try:
                    with st.spinner("Loading model..."):
                        model = load_model(model_path)
                        st.session_state['model'] = model
                        st.session_state['model_loaded'] = True
                    
                    st.markdown('<div class="success-box">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
                    st.write(f"**Model Parameters:** {model.count_params():,}")
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            else:
                st.error(f"File not found: {model_path}")
        
        # Option 2: Upload model file
        st.subheader("Option 2: Upload Model File")
        uploaded_model = st.file_uploader("Choose H5 model file", type=['h5'])
        
        if uploaded_model is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    tmp_file.write(uploaded_model.read())
                    temp_model_path = tmp_file.name
                
                with st.spinner("Loading uploaded model..."):
                    model = load_model(temp_model_path)
                    st.session_state['model'] = model
                    st.session_state['model_loaded'] = True
                
                st.markdown('<div class="success-box">✅ Model uploaded and loaded!</div>', unsafe_allow_html=True)
                st.write(f"**Model Parameters:** {model.count_params():,}")
                
                # Clean up
                os.unlink(temp_model_path)
                
            except Exception as e:
                st.error(f"Error loading uploaded model: {str(e)}")
    
    # Right column - File upload and analysis
    with col2:
        st.header("📁 Brain Scan Analysis")
        
        # Check if model is loaded
        if 'model_loaded' not in st.session_state:
            st.info("👈 Please load a model first")
        else:
            st.success("Model is ready!")
            
            # File upload
            uploaded_scan = st.file_uploader("Upload brain scan (.nii or .nii.gz)", type=['nii', 'gz'])
            
            if uploaded_scan is not None:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
                        tmp_file.write(uploaded_scan.read())
                        temp_scan_path = tmp_file.name
                    
                    # Load and display file info
                    st.write("**File Info:**")
                    st.write(f"- Name: {uploaded_scan.name}")
                    st.write(f"- Size: {len(uploaded_scan.getvalue())} bytes")
                    
                    # Load the brain scan
                    with st.spinner("Loading brain scan..."):
                        brain_volume = load_nifti_file(temp_scan_path)
                    
                    if brain_volume is not None:
                        # Display volume information
                        st.write("**Volume Information:**")
                        st.write(f"- Shape: {brain_volume.shape}")
                        st.write(f"- Data type: {brain_volume.dtype}")
                        st.write(f"- Min value: {np.min(brain_volume):.2f}")
                        st.write(f"- Max value: {np.max(brain_volume):.2f}")
                        
                        # Preprocess for model
                        with st.spinner("Preprocessing..."):
                            processed_volume = preprocess_volume(brain_volume)
                        
                        if processed_volume is not None:
                            # Make prediction
                            with st.spinner("Running AI analysis..."):
                                model = st.session_state['model']
                                predictions = model.predict(processed_volume, verbose=0)
                                predicted_class = np.argmax(predictions[0])
                                confidence = float(np.max(predictions[0]))
                            
                            # Display results
                            st.header("🎯 Analysis Results")
                            
                            # Class names (adjust these based on your model)
                            class_names = ['MCI', 'NC']  # Modify as needed
                            predicted_label = class_names[predicted_class] if predicted_class < len(class_names) else f"Class_{predicted_class}"
                            
                            # Results display
                            if predicted_label == 'NC':
                                st.markdown(f'<div class="success-box"><h3>✅ Prediction: {predicted_label}</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                            elif predicted_label == 'MCI':
                                st.markdown(f'<div class="warning-box"><h3>⚠️ Prediction: {predicted_label}</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="danger-box"><h3>🚨 Prediction: {predicted_label}</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                            
                            # Probability chart
                            if len(predictions[0]) <= len(class_names):
                                prob_data = {
                                    'Class': class_names[:len(predictions[0])],
                                    'Probability': predictions[0][:len(class_names)]
                                }
                                prob_df = pd.DataFrame(prob_data)
                                
                                fig = px.bar(prob_df, x='Class', y='Probability', 
                                           title='Classification Probabilities',
                                           color='Probability',
                                           color_continuous_scale='viridis')
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Clean up temp file
                    os.unlink(temp_scan_path)
                    
                except Exception as e:
                    st.error(f"Error processing brain scan: {str(e)}")
                    st.exception(e)
    
    # Visualization section
    if 'model_loaded' in st.session_state and uploaded_scan is not None and brain_volume is not None:
        st.header("📊 Brain Scan Visualization")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["🔍 Slice View", "📊 Volume Stats"])
        
        with tab1:
            # Create slice visualization
            fig = create_slice_visualization(brain_volume, f"Brain Scan: {uploaded_scan.name}")
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        
        with tab2:
            # Volume statistics
            st.subheader("Volume Statistics")
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.metric("Mean Intensity", f"{np.mean(brain_volume):.2f}")
                st.metric("Standard Deviation", f"{np.std(brain_volume):.2f}")
                st.metric("Volume Shape", f"{brain_volume.shape}")
            
            with stats_col2:
                st.metric("Min Intensity", f"{np.min(brain_volume):.2f}")
                st.metric("Max Intensity", f"{np.max(brain_volume):.2f}")
                st.metric("Non-zero Voxels", f"{np.count_nonzero(brain_volume):,}")
            
            # Intensity distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(brain_volume.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Intensity Distribution')
            ax.set_xlabel('Intensity Value')
            ax.set_ylabel('Frequency')
            st.pyplot(fig, use_container_width=True)
    
    # Instructions
    st.markdown("---")
    st.header("📋 Instructions")
    
    instructions_col1, instructions_col2 = st.columns(2)
    
    with instructions_col1:
        st.markdown("""
        **Step 1: Load Model**
        - Enter the path to your H5 model file
        - Or upload the model file directly
        - Wait for confirmation message
        """)
    
    with instructions_col2:
        st.markdown("""
        **Step 2: Analyze Brain Scan**
        - Upload a .nii or .nii.gz brain scan file
        - Wait for processing to complete
        - View results and visualizations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🧠 Alzheimer's Disease 3D CNN Classifier**
    
    ⚕️ **Medical Disclaimer**: This tool is for research purposes only. 
    Consult healthcare professionals for medical diagnosis.
    """)

if __name__ == "__main__":
    main()
