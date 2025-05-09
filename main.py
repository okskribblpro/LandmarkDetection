import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# --- Pagina-opmaak ---
st.set_page_config(page_title="Beeldherkenning met AI", layout="wide")
st.image("thomasmorelogo.png", width=200)
st.title("Image recognition and comparing")
st.markdown("Do More with our new A.I. landmark recognition tool.")

# --- Model en labels laden ---
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# --- Functie om voorspelling te doen ---
def predict_image(uploaded_image):
    size = (224, 224)
    image = ImageOps.fit(uploaded_image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# --- Layout in 2 kolommen ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("📷 Upload afbeelding 1", type=["jpg", "jpeg", "png"], key="file1")
    if uploaded_file1:
        image1 = Image.open(uploaded_file1).convert("RGB")
        st.image(image1, caption="Afbeelding 1", use_container_width=True)

with col2:
    uploaded_file2 = st.file_uploader("📷 Upload afbeelding 2", type=["jpg", "jpeg", "png"], key="file2")
    if uploaded_file2:
        image2 = Image.open(uploaded_file2).convert("RGB")
        st.image(image2, caption="Afbeelding 2", use_container_width=True)

# --- AI-vergelijking uitvoeren ---
if uploaded_file1 and uploaded_file2:
    st.markdown("---")
    st.subheader("🔍 AI Analyse Resultaten")

    class1, conf1 = predict_image(image1)
    st.markdown(f"**Afbeelding 1 voorspelling:** {class1} (🔍 {conf1*100:.2f}%)")

    class2, conf2 = predict_image(image2)
    st.markdown(f"**Afbeelding 2 voorspelling:** {class2} (🔍 {conf2*100:.2f}%)")

import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Webcam AI-classificatie
st.markdown("---")
st.subheader("📸 Live Webcam Classificatie")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Voorspelling
        class_name, confidence_score = predict_image(pil_image)

        # Tekst op frame tekenen
        label = f"{class_name} ({confidence_score*100:.1f}%)"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start webcam feed
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)


# --- Footer ---
st.markdown("---")
st.caption("🎓 Gemaakt tijdens stage 2025")
