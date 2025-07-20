import cv2
import time
import supervision as sv
from ultralytics import YOLO

VIDEO_PATH = "files/person2.mp4"  
MODEL_NAME = "yolov8n.pt"     
CONFIDENCE_THRESHOLD = 0.3    
IOU_THRESHOLD = 0.5      

def main():

    print("Starting object tracking pipeline...")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    model = YOLO(MODEL_NAME)
    print("YOLO model loaded successfully.")

    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE)
    
    print("Tracker and annotators initialized.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        start_time = time.time()

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.class_id == 0]

        tracked_objects = tracker.update_with_detections(detections=detections)

        labels = [
            f"ID:{tracker_id} {model.model.names[class_id]}"
            for class_id, tracker_id
            in zip(tracked_objects.class_id, tracked_objects.tracker_id)
        ]

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_objects)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_objects, labels=labels)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Object Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()   
    print("Pipeline finished.")

if __name__ == "__main__":
    main()