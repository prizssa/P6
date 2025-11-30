# Import YOLO from ultralytics
from ultralytics import YOLO

# Import OpenCV and Mathq
import cv2
import math

# Load rock-paper-scissors model
model = YOLO("content/runs/detect/train/weights/bestv.pt")

# Load the model's labels and store them as class_names
class_names = model.names

# Capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()
    # Quit when no more frame
    if not ret:
        break

    # Run object detector
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls = int(box.cls[0])
            label = class_names[cls]
            conf = math.ceil(box.conf[0] * 100) / 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Under cv.putText, adjust the font size to 0.7
            cv2.putText(frame,
                        f"{label} {conf}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

    # Display the resulting frame
    cv2.imshow("Webcam Detection", frame)

    # stop when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
webcam.release()
cv2.destroyAllWindows()