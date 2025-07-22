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
