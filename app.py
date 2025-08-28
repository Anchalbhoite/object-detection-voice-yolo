import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from gtts import gTTS
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🔍 Object Detection with Voice (YOLOv8)")
st.markdown("Upload an image to detect objects. The app will also announce detected objects with speech.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Read image
    img = cv2.imread(tfile.name)

    # Run YOLO
    results = model(img)
    annotated_img = results[0].plot()

    # Display results
    st.image(annotated_img, channels="BGR", caption="Detected Objects")

    # Extract detected objects
    names = results[0].names
    detected = [names[int(cls)] for cls in results[0].boxes.cls]

    if detected:
        st.success(f"Detected: {', '.join(detected)}")

        # Convert detected objects into speech
        text_to_speak = " , ".join(detected)
        tts = gTTS(text=text_to_speak, lang="en")
        audio_path = "detected.mp3"
        tts.save(audio_path)

        # Play audio in Streamlit
        st.audio(audio_path, format="audio/mp3")
