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
