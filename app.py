import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("models/final_model.h5")

CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
IMG_SIZE = (224, 224)

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to predict tumor type")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction")
    st.write(f"🧠 Tumor Type: **{predicted_class}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
