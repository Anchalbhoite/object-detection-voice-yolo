
# Object Detection and Recognition with Voice using YOLOv8

# 🔍 Object Detection with Voice (YOLOv8)  

[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)  
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow?logo=python)](https://www.python.org/)  
[![Deployment](https://img.shields.io/badge/Deployed%20on-Streamlit%20Cloud-brightgreen?logo=streamlit)](https://object-detection-voice-yolo-wv.streamlit.app/)  

🚀 **Live Demo**: 👉 [Object Detection App](https://object-detection-voice-yolo-wv.streamlit.app/)  

---

## 📌 Project Overview  
This project demonstrates **real-time object detection** using **YOLOv8** with added **voice feedback**.  
Users can upload an image or use their **webcam camera** to detect objects, and the detected items are announced using **Google Text-to-Speech (gTTS)**.  

✅ Perfect for showcasing **Computer Vision + AI deployment** skills.  

---

## ✨ Features
- 🖼️ **Image Upload** & 📸 **Webcam Capture** support  
- ⚡ **YOLOv8 (Ultralytics)** for accurate object detection  
- 🔊 **Voice feedback** using Google TTS (gTTS)  
- 🌐 **Streamlit Cloud deployment** (accessible on any device)  
- 🟢 Lightweight (uses `opencv-python-headless`)  

---

## 🛠 Tech Stack
- **Python**  
- **YOLOv8 (Ultralytics)** – Object detection  
- **OpenCV (Headless)** – Image processing  
- **gTTS (Google Text-to-Speech)** – Voice output  
- **Streamlit** – Web app frontend  
- **Streamlit Cloud** – Deployment  

---

## 📷 Screenshots
### Upload Image  
![Upload Demo](images/screenshot_upload.png)  

### Camera Input  
![Camera Demo](images/screenshot_camera.png)  

---

## ⚙️ Installation (Run Locally)
```bash
# Clone the repo
git clone https://github.com/your-username/object-detection-voice-yolo.git
cd object-detection-voice-yolo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

