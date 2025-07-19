import cv2
import time
import os
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  

video_path = "C:\\Users\\shaya\\Downloads\\videos\\person2.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = os.path.join(os.path.dirname(video_path), "output_with_stats.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model(frame)
    annotated_frame = results[0].plot()

    end_time = time.time()
    elapsed = end_time - start_time
    fps_display = 1 / (elapsed + 1e-8)

    num_detections = len(results[0].boxes)

    cv2.putText(annotated_frame, f"FPS: {fps_display:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(annotated_frame, f"Detections: {num_detections}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(annotated_frame, f"Time: {elapsed:.2f}s", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('YOLOv8 Detection - Press Q to Quit', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()