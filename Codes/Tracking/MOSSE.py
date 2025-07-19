import cv2
import time
import os
from ultralytics import YOLO

def create_tracker():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerMOSSE_create()
    else:
        return cv2.TrackerMOSSE_create()

def run_yolo_mosse(video_source=0, detection_interval=30, conf_threshold=0.3, output_path="output_mosse.mp4"):
    cap = cv2.VideoCapture(video_source)
    model = YOLO("yolov5s.pt")

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    # Read the first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    # Get video properties
    frame_height, frame_width = frame.shape[:2]
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps == 0 or input_fps != input_fps:  # handle NaN
        input_fps = 25

    # Define VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, input_fps, (frame_width, frame_height))

    # Re-initialize video after reading first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, labels[i], (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Lost {labels[i]}", (10, 30 + 30 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time

        fps = frame_count / total_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show and save the frame
        cv2.imshow("YOLO + MOSSE", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to: {os.path.abspath(output_path)}")

# 🔁 To run with webcam (0) or video file path:
# run_yolo_mosse(0)
# run_yolo_mosse("files/person1.mp4")
video_path = "files/person3.mp4"
run_yolo_mosse(0)  # Change this to a file path if needed
