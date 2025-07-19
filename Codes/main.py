import cv2
from ultralytics import YOLO

def create_tracker(tracker_type):
    if hasattr(cv2, 'legacy'):
        if tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        elif tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif tracker_type == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
    else:
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        elif tracker_type == "MOSSE":
            return cv2.TrackerMOSSE_create()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0

def run_tracker(tracker_type, detection_interval=30):
    print(f"Running {tracker_type} Tracker on full video with detection every {detection_interval} frames")
    cap = cv2.VideoCapture("files/person1.MP4")
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    model = YOLO("yolov5s.pt")
    
    results = model.predict(source=frame, conf=0.3)
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No objects detected on first frame.")
        cap.release()
        return

    trackers = []
    labels = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2 - x1, y2 - y1)
        label = model.names[int(box.cls[0])]
        tracker = create_tracker(tracker_type)
        tracker.init(frame, bbox)
        trackers.append(tracker)
        labels.append(label)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        display_frame = frame.copy()

        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = tuple(map(int, box))
                color = (0, 255, 0) if tracker_type == "CSRT" else (255, 0, 0) if tracker_type == "KCF" else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, labels[i], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display_frame, f"Lost {labels[i]}", (20, 20 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if frame_count % detection_interval == 0:
            results = model.predict(source=frame, conf=0.3)
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                label = model.names[int(box.cls[0])]

                overlaps = False
                for tracker in trackers:
                    success, tracked_box = tracker.update(frame)
                    if not success:
                        continue
                    if iou(bbox, tuple(map(int, tracked_box))) > 0.3:
                        overlaps = True
                        break
                
                if not overlaps:
                    new_tracker = create_tracker(tracker_type)
                    new_tracker.init(frame, bbox)
                    trackers.append(new_tracker)
                    labels.append(label)

        cv2.putText(display_frame, f"{tracker_type} Tracker", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if tracker_type == "CSRT" else (255, 0, 0) if tracker_type == "KCF" else (0, 0, 255), 2)
        cv2.imshow(f"{tracker_type} Tracker", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

run_tracker("MOSSE", detection_interval=30)
run_tracker("KCF", detection_interval=30)
run_tracker("CSRT", detection_interval=30)
