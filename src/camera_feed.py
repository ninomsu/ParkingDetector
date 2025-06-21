import cv2
import torch
import time
import os
import sys
from pushover import Client
import xml.etree.ElementTree as ET
import logging
from datetime import datetime
import atexit

# === YOLO model and constants ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', trust_repo=True)
model.conf = 0.3
CONF_THRESHOLD = 0.4
LEFT_ROI = (230, 270, 140, 150)  # ROI for far-left parking spot
RIGHT_ROI = (760, 300, 100, 150)  # ROI for far-left parking spot

EMPTY = "Empty"
TAKEN = "Taken"
ERROR = "Error"

def boxes_intersect(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def get_state_with_extra_verification():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Failed to capture frame")
        return ERROR, None

    results = model(frame)
    detections = results.xyxy[0]
    spot_taken = False

    for *box, conf, cls in detections:
        if int(cls) in [2, 7]:
            if float(conf) >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                car_box = (x1, y1, x2 - x1, y2 - y1)
                if boxes_intersect(car_box, LEFT_ROI):
                    print(f"Detected class: {int(cls)}, confidence: {conf.item():.2f}")
                    spot_taken = True
                    break

    state = TAKEN if spot_taken else EMPTY
    drawn_frame = draw_detections(frame, state, detections)
    
    return state, drawn_frame

def draw_detections(frame, state, detections):
    for *box, conf, cls in detections:
        if float(conf) >= CONF_THRESHOLD and int(cls) in [2, 7]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    x, y, w, h = LEFT_ROI
    color = (0, 0, 255) if state == TAKEN else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, state, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    x, y, w, h = RIGHT_ROI
    color = (0, 0, 255) if state == TAKEN else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, state, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Left Spot Detection", frame)
    cv2.waitKey(1)
    return frame

# === Main loop ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Warming up camera...")
time.sleep(2)

while True:
    current_state, frame = get_state_with_extra_verification()
    if current_state == ERROR:
        log("Error getting state, ending...")
        break

cap.release()
cv2.destroyAllWindows()
logging.shutdown()