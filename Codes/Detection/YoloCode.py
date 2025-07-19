import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov5s.pt")

video_path = 0  # use 0 for webcam
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model(frame)

    annotated_frame = results[0].plot()

    end_time = time.time()
    fps_display = 1 / (end_time - start_time + 1e-8)

    cv2.putText(annotated_frame,
                f"FPS: {fps_display:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow('YOLO Detection', annotated_frame)

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
