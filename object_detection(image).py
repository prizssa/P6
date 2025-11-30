# Import YOLO
from ultralytics import YOLO

# Import OpenCV and Math
import cv2
import math

# Load rock-paper-scissors model
model = YOLO(model="content/runs/detect/train/weights/best.pt")

# Load the model's labels and store them as class_names
class_names = model.names

# Load image from 'image1.jpg' and store it as img
img = cv2.imread("image1.jpg")

# Resize the image to dimension 1280x720
img = cv2.resize(img, (1280, 720))

# Detect objects in the image and store them as results
results = model(img)

# Draw the bounding boxes and labels on the image
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = math.ceil(box.conf[0] * 100) / 100
        label = f"{class_names[cls]} {conf}"

        # box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        # text (font size MUST be 1.0 like specified)
        cv2.putText(img, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# Show the image
cv2.imshow("RPS Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()