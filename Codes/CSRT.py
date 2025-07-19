import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov5s.pt")

video_path = "files/person3.mp4"  # or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Create CSRT Tracker function
def create_csrt_tracker():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        return cv2.TrackerCSRT_create()

# Check if video loaded
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Get video properties
frame_height, frame_width = frame.shape[:2]
fps_input = cap.get(cv2.CAP_PROP_FPS)
if fps_input == 0:  # fallback in case FPS can't be read
    fps_input = 25

# Create VideoWriter to save output
output_path = "output_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI
out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

# Initial detection with YOLO
results = model.predict(frame, conf=0.5)
trackers = []
labels = []

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tracker = create_csrt_tracker()
        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        trackers.append(tracker)
        labels.append(label)

prev_time = time.time()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for i, tracker in enumerate(trackers):
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, labels[i], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show and save the frame
    cv2.imshow("YOLO + CSRT Tracker", frame)
    out.write(frame)  # ✅ Save the current frame

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

# Cleanup
cap.release()
out.release()  # ✅ Release the writer
cv2.destroyAllWindows()
