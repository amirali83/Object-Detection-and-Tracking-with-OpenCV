import cv2
from ultralytics import YOLO

def detect_objects_in_video(video_path, model_path="yolov5s.pt", conf_threshold=0.3):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf_threshold)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(30) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

detect_objects_in_video("C:/Users/shaya/Downloads/videos/person2.MP4")
