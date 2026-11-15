import cv2
import numpy as np
from ultralytics import YOLO
import itertools
from scipy.optimize import linear_sum_assignment

model = YOLO("yolov5s.pt")

cap = cv2.VideoCapture("files/person2.mp4")

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

IOU_THRESHOLD = 0.3

class TrackedObject:
    _id_iter = itertools.count()

    def __init__(self, box, label, gray):
        self.id = next(self._id_iter)
        self.label = label
        self.box = box
        self.lost_frames = 0
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        x1, y1, x2, y2 = box
        roi_gray = gray[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
        if corners is not None:
            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
        self.prev_pts = corners
        self.prev_gray = gray.copy()
        self.box_size = (x2 - x1, y2 - y1)
        if self.prev_pts is not None:
            center = np.mean(self.prev_pts, axis=0).reshape(2, 1)
            self.kalman.statePre[:2] = center
            self.kalman.statePre[2:] = 0
            self.kalman.statePost = self.kalman.statePre.copy()

    def update(self, frame, gray):
        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.lost_frames += 1
            return
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **lk_params)
        good_new = next_pts[status == 1]
        if len(good_new) == 0:
            self.lost_frames += 1
            return
        center = np.mean(good_new, axis=0).reshape(2, 1)
        self.kalman.correct(center)
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray.copy()
        self.lost_frames = 0

    def predict(self):
        return self.kalman.predict()

    def draw(self, frame):
        pred = self.predict()
        pred_x, pred_y = int(pred[0]), int(pred[1])
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            pts = self.prev_pts.reshape(-1, 2)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            padding = 10
            top_left = (int(x_min - padding), int(y_min - padding))
            bottom_right = (int(x_max + padding), int(y_max + padding))
        else:
            w, h = self.box_size
            top_left = (int(pred_x - w / 2), int(pred_y - h / 2))
            bottom_right = (int(pred_x + w / 2), int(pred_y + h / 2))
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame, f"{self.label}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self.box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

tracked_objects = []
frame_count = 0
REDETECT_INTERVAL = 15
MAX_LOST_FRAMES = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for obj in tracked_objects:
        obj.update(frame, gray)

    if frame_count % REDETECT_INTERVAL == 0:
        results = model(frame)[0]
        detections = []
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls_id = int(det.cls[0])
            label = model.names[cls_id]
            detections.append(((x1, y1, x2, y2), label))

        cost_matrix = np.zeros((len(tracked_objects), len(detections)), dtype=np.float32)
        for i, obj in enumerate(tracked_objects):
            for j, (det_box, _) in enumerate(detections):
                cost_matrix[i, j] = 1 - compute_iou(obj.box, det_box)

        matched_rows, matched_cols = linear_sum_assignment(cost_matrix)
        unmatched_detections = set(range(len(detections)))
        matched_ids = set()

        for r, c in zip(matched_rows, matched_cols):
            if cost_matrix[r, c] < 1 - IOU_THRESHOLD:
                obj = tracked_objects[r]
                det_box, label = detections[c]
                obj.__init__(det_box, label, gray)
                matched_ids.add(r)
                unmatched_detections.discard(c)

        for idx in unmatched_detections:
            det_box, label = detections[idx]
            tracked_objects.append(TrackedObject(det_box, label, gray))

    tracked_objects = [obj for obj in tracked_objects if obj.lost_frames < MAX_LOST_FRAMES]

    for obj in tracked_objects:
        obj.draw(frame)

    cv2.imshow("Multi-Object Tracker", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
