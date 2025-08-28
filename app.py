import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🔍 Object Detection with Voice (YOLOv8)")
st.markdown("This app detects objects in real-time and announces their names with speech.")

run = st.checkbox("Start Camera")

if run:
    cap = cv2.VideoCapture(0)

    # Text-to-speech engine
    engine = pyttsx3.init()

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model
        results = model(frame)

        # Plot results
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")

        # Get detected objects
        names = results[0].names
        detected = [names[int(cls)] for cls in results[0].boxes.cls]

        if detected:
            for obj in detected:
                engine.say(obj)
            engine.runAndWait()
