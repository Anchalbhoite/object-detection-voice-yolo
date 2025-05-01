import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech (optional)
engine.setProperty('volume', 1.0)  # Volume (optional)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can also use 'yolov8s.pt' or custom weights

# Open the webcam
cap = cv2.VideoCapture(0)

# To store already spoken labels to avoid repeating
spoken_labels = set()

# For calculating FPS
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Predict using YOLO
    results = model.predict(source=frame, save=False, verbose=False)

    # Draw the annotations
    annotated_frame = results[0].plot()

    # Detect objects
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Class id
            label = model.names[cls_id]  # Class label
            
            # Speak only once for each object detected
            if label not in spoken_labels:
                spoken_labels.add(label)
                print(f"Detected: {label}")
                engine.say(f"{label} detected")
                engine.runAndWait()

    # Calculate FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time + 1e-5))  # avoid division by zero
    prev_frame_time = new_frame_time

    # Display FPS on frame
    cv2.putText(annotated_frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("YOLOv8 Detection with Voice", annotated_frame)

    # Press 'r' to reset spoken labels (start announcing again)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        spoken_labels.clear()
        print("Spoken labels reset.")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
