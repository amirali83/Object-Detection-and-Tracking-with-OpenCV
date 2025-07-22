import cv2
import time
from ultralytics import YOLO

def create_tracker():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerKCF_create()
    else:
        return cv2.TrackerKCF_create()

def run_yolo_kcf(video_source=0, detection_interval=30, conf_threshold=0.3, output_file="output.avi"):
    cap = cv2.VideoCapture(video_source)
    
    model = YOLO("yolov5s.pt")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0  
    

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(output_file, fourcc, fps_input, (width, height))

    

    trackers = []
    labels = []
    frame_count = 0
    total_time = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % detection_interval == 1 or len(trackers) == 0:
            results = model.predict(source=frame, conf=conf_threshold)
            detections = results[0].boxes

            trackers = []
            labels = []

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                label = model.names[int(box.cls[0])]

                tracker = create_tracker()
                tracker.init(frame, bbox)
                trackers.append(tracker)
                labels.append(label)


            else:
                for i, tracker in enumerate(trackers):
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = tuple(map(int, bbox))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, labels[i], (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"Lost {labels[i]}", (10, 30 + 30 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time

        fps = frame_count / total_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("YOLO + KCF", frame)
        out.write(frame)  

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    out.release()  
    cv2.destroyAllWindows()


run_yolo_kcf(video_source="C:\\Users\\shaya\\Downloads\\videos\\person1.mp4", output_file="output.mp4")
