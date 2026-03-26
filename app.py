import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import gdown
import os

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = 'plant_disease_model.h5'
    if not os.path.exists(model_path):
        file_id = '1CeYtagcTsqnAWyKRaBO_43uG32bkgMKA'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()
IMG_SIZE = 64

st.title("🌿 Plant Disease Detector")
st.markdown("Upload a **plant leaf image** to detect disease instantly.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)
    st.markdown("---")

    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing leaf..."):
            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            top3_idx = np.argsort(predictions[0])[::-1][:3]
            predicted_class = class_names[top3_idx[0]]
            confidence = round(100 * predictions[0][top3_idx[0]], 2)
            display_name = predicted_class.replace(
                '___', ' → ').replace('_', ' ')

            st.markdown("## Detection Result")
            if confidence > 85:
                st.success(f"**Disease: {display_name}**")
            else:
                st.warning(f"**Disease: {display_name}** (Low Confidence)")

            st.metric("Confidence Score", f"{confidence}%")

            st.markdown("### Top 3 Predictions")
            for i, idx in enumerate(top3_idx):
                name = class_names[idx].replace(
                    '___', ' → ').replace('_', ' ')
                conf = round(100 * predictions[0][idx], 2)
                st.progress(int(conf), text=f"{i+1}. {name} → {conf}%")

            st.markdown("### Plant Health Status")
            if 'healthy' in predicted_class.lower():
                st.success("This plant is **Healthy!**")
            else:
                st.error("This plant is **Diseased!**")
                st.info("Please consult an agronomist for treatment.")

st.markdown("---")
st.markdown(
    "<center>🌱 Built with CNN + TensorFlow + Streamlit</center>",
    unsafe_allow_html=True
)