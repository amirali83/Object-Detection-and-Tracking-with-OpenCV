import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO("yolov5s.pt")

source = "files/person1.mp4"
cap = cv2.VideoCapture(source)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

prev_gray = None
prev_pts = None
tracking = False
object_center = None
object_label = None

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracking and prev_pts is not None:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        if len(good_new) > 0:
            object_center = np.mean(good_new, axis=0).reshape(2, 1)
            kalman.correct(object_center)

            prev_pts = good_new.reshape(-1, 1, 2)
            prev_gray = gray.copy()
        else:
            tracking = False

    if not tracking:
        results = model(frame)[0]
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls[0])
            label = model.names[cls]

            roi_gray = gray[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

            if corners is not None:
                corners[:, 0, 0] += x1
                corners[:, 0, 1] += y1
                prev_pts = corners.astype(np.float32)
                prev_gray = gray.copy()
                tracking = True

                object_center = np.mean(prev_pts, axis=0).reshape(2, 1)
                kalman.statePre[:2] = object_center
                kalman.statePre[2:] = 0
                kalman.statePost = kalman.statePre.copy()

                object_label = label
                break

    predicted = kalman.predict()
    pred_x, pred_y = int(predicted[0]), int(predicted[1])

    if tracking and prev_pts is not None and len(prev_pts) >= 3:
        x_coords = prev_pts[:, 0, 0]
        y_coords = prev_pts[:, 0, 1]
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))

        width = max_x - min_x
        height = max_y - min_y

        top_left = (int(pred_x - width / 2), int(pred_y - height / 2))
        bottom_right = (int(pred_x + width / 2), int(pred_y + height / 2))

        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        if object_label:
            cv2.putText(frame, object_label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLO + Optical Flow + Kalman + Adaptive Box", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
