from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame)

    # Get annotated frame with detections
    annotated_frame = results[0].plot()

    # Resize frames to same height for side-by-side display
    height = 480
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    annotated_resized = cv2.resize(annotated_frame, (int(annotated_frame.shape[1] * height / annotated_frame.shape[0]), height))

    # Concatenate original and annotated frames horizontally
    combined = np.hstack((frame_resized, annotated_resized))

    # Show combined frame
    cv2.imshow('Original (Left) | YOLOv8 Detection (Right)', combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
