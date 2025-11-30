
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import math
import os

st.title("Rock Paper Scissors Detector (Webcam)")

# Use Streamlit's camera input widget
camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    # Convert camera image to numpy array for OpenCV
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None:
        st.error("Could not read the image. Please try again.")
    else:
        img = cv2.resize(img, (1280, 720))

        # Load YOLO model
        model_path = "content/runs/detect/train/weights/best.pt" 
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found.")
        else:
            model = YOLO(model_path)
            class_names = model.names

            # Run detection
            results = model(img)

            # Draw bounding boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    label = f'{class_names[cls]} {conf}'
                    cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Convert BGR to RGB for display
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)