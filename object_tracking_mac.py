import cv2
import numpy as np
from math import sqrt

# Initialize Haar Cascade (fallback for face detection on macOS)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible. Check macOS camera permissions.")
    exit()

count = 0
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (fallback logic)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    center_points2 = []

    for (x, y, w, h) in faces:
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        center_points2.append((cx, cy))
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 200), 2)

    if count == 0:
        for pt in center_points2:
            tracking_objects[track_id] = pt
            track_id += 1
    else:
        for object_id, pt in tracking_objects.copy().items():
            object_exists = False
            for pt2 in center_points2.copy():
                euc = sqrt((pt[0] - pt2[0]) ** 2 + (pt[1] - pt2[1]) ** 2)
                if euc < 20:
                    tracking_objects[object_id] = pt2
                    object_exists = True
                    center_points2.remove(pt2)
                    break
            if not object_exists:
                tracking_objects.pop(object_id)
        for pt in center_points2:
            tracking_objects[track_id] = pt
            track_id += 1

    count += 1
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (255, 255, 0), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
