import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile

# Set page config
st.set_page_config(
    page_title="Coral reef monitoring model ",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("Coral Reef Monitoring Model")
st.markdown("Upload an image to detect the health of coral reefs ")

# Load the model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Sidebar for model selection and confidence threshold
with st.sidebar:
    st.header("Settings")
    model_path = "best.pt"  
    confidence = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    
    # Load model button
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            try:
                model = load_model(model_path)
                st.success(f"Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")

# Main content
try:
    model = load_model(model_path)
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Display uploaded image and predictions
    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Run inference and get results
        with st.spinner("Detecting objects..."):
            # Save uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_filename = temp_file.name
                image.save(temp_filename)
            
            # Run prediction
            results = model.predict(source=temp_filename, conf=confidence)
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            # Display results
            with col2:
                st.subheader("Detection Results")
                
                # Get the first result and its plot
                if len(results) > 0:
                    # Display image with bounding boxes
                    result_plot = results[0].plot()
                    st.image(result_plot, channels="BGR", use_container_width=True)
                    
                    # Get and display detection details
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.subheader("Detected Objects")
                        
                        # Create a table of detections
                        data = []
                        for box in boxes:
                            class_id = int(box.cls[0].item())
                            class_name = results[0].names[class_id]
                            confidence = round(box.conf[0].item(), 2)
                            data.append({"Class": class_name, "Confidence": confidence})
                        
                        st.table(data)
                    else:
                        st.info("Bleached Coral reefs detected")
                else:
                    st.info("No results returned from model.")
                
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please make sure you have loaded a valid  model.")

# Footer
st.markdown("---")
