import cv2
from ultralytics import YOLO

model = YOLO('./models/yolo11n-seg.pt')
cap = cv2.VideoCapture("your_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    # Visualize the segmentation masks
    annotated_frame = results.plot()

    cv2.imshow("YOLOv8 Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
