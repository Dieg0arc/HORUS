

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

model = YOLO('/home/srm/Documents/hands-pose-estimation/runs/pose/train/weights/best.pt')

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    result = model.predict(frame)

    cv2.imshow('Webcam Video Stream', result[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

