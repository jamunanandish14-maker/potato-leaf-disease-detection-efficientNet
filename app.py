# ==========================================
# app.py
# POTATO LEAF DISEASE DETECTION
# USING EFFICIENTNETB0
# ==========================================

import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# PAGE TITLE
# ==========================================

st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image to detect disease using EfficientNetB0.")

# ==========================================
# LOAD MODEL
# ==========================================
model = load_model("potato_efficientnet.keras")

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Choose a potato leaf image...",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Display image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    img = img.resize((224,224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # ==========================================
    # RESULTS
    # ==========================================

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence Score: {confidence:.2f}%")
