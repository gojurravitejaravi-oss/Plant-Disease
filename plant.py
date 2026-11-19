import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱")
st.title("🌱 Plant Disease Detector")
st.write("Upload a leaf photo and I will tell you the disease.")

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("plant_model.h5", compile=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_my_model()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Leaf", use_container_width=True)

    if st.button("Predict Disease"):
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        disease_name = class_names[predicted_class]

        st.success(f"Prediction: **{disease_name}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        if "healthy" in disease_name.lower():
            st.balloons()
            st.write("✅ Your plant is Healthy!")
        else:
            st.warning("⚠️ Disease Detected!")
else:
    st.write("👆 Please upload an image to start.")
