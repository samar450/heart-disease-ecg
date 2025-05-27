import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import os
import random

model = tf.keras.models.load_model("models/heart_disease_model_binary.keras")
class_names = ["Normal", "Abnormal"]

normal_dir = "valid_binary/Normal"
abnormal_dir = "valid_binary/Abnormal"

# UI setup
st.set_page_config(page_title="ECG Image Classifier", layout="centered")
st.title("ECG Image Classifier")
st.markdown("Upload an ECG image or use a random sample from the validation set to classify it as **Normal** or **Abnormal**.")

col1, col2 = st.columns([1, 1])
uploaded_file = col1.file_uploader("Upload ECG image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
use_sample = col2.button("Use Random Sample Image")

if use_sample:
    # Pick randomly from either class
    selected_dir = random.choice([normal_dir, abnormal_dir])
    image_files = [f for f in os.listdir(selected_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if image_files:
        sample_image_path = os.path.join(selected_dir, random.choice(image_files))
        uploaded_file = open(sample_image_path, "rb")
        st.info(f"Random sample selected from: {selected_dir}")
    else:
        st.error(f"No images found in {selected_dir}.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ECG Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    start_time = time.time()
    prediction = model.predict(img_array)[0][0]
    end_time = time.time()
    elapsed = (end_time - start_time) * 1000

    pred_class = class_names[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader("Prediction Result")
    st.markdown(f"**{pred_class}** with confidence **{confidence*100:.2f}%**")
    st.caption(f"Inference time: {elapsed:.2f} ms")
