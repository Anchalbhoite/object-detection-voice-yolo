import streamlit as st
import cv2
from ultralytics import YOLO
from gtts import gTTS
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("🔍 Object Detection with Voice (YOLOv8)")
st.markdown("Upload an image **or use your camera**. The app will detect objects and speak them out.")

# Choose input method
option = st.radio("Choose input method:", ("📂 Upload Image", "📸 Use Camera"))

img = None

# File upload option
if option == "📂 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        img = cv2.imread(tfile.name)

# Camera option
elif option == "📸 Use Camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(camera_file.getbuffer())
        img = cv2.imread(tfile.name)

# Process the image if available
if img is not None:
    # Run YOLO
    results = model(img)
    annotated_img = results[0].plot()

    # Display annotated image
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
