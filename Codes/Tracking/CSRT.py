import cv2
import time
from ultralytics import YOLO


model = YOLO("yolov5s.pt")

video_path = "C:\\Users\\shaya\\Downloads\\videos\\person2.mp4"  
cap = cv2.VideoCapture(video_path)

def create_csrt_tracker():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        return cv2.TrackerCSRT_create()

ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

frame_height, frame_width = frame.shape[:2]
fps_input = cap.get(cv2.CAP_PROP_FPS)
if fps_input == 0:
    fps_input = 25

output_path = "output_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

trackers = []
labels = []
frame_count = 0
detect_interval = 30  

prev_time = time.time()
