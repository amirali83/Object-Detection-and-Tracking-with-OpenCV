import cv2
import time
import os
from ultralytics import YOLO

model = YOLO("yolov8s.pt") 
video_path = "C:\\Users\\shaya\\Downloads\\videos\\person2.mp4"
cap = cv2.VideoCapture(video_path)
