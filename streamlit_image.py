import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import math

st.title("Rock Paper Scissors Detector")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Load YOLO model
model = YOLO("content/runs/detect/train/weights/best.pt")
class_names = model.names

if uploaded_file is not None:
    # Convert uploaded image into OpenCV format
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize
    img = cv2.resize(img, (1280, 720))

    # Run detection
    results = model(img)

    # Draw boxes + labels
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            conf = math.ceil(box.conf[0] * 100) / 100

            label = f"{class_names[cls]} {conf}"

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 3)
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

    # Convert BGR → RGB for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show result
    st.image(img_rgb, caption="Detection Result", use_column_width=True)