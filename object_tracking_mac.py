import cv2
import numpy as np
from math import sqrt

# Initialize Haar Cascade for face detection (fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kalman Filter Setup for better tracking
def create_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
    return kalman

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible. Check macOS camera permissions.")
    exit()

count = 0
tracking_objects = {}
track_id = 0
kalman_filters = {}  # Store Kalman Filters per object

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade fallback method
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    center_points2 = []
    for (x, y, w, h) in faces:
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        center_points2.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 200), 2)

    if count == 0:
        for pt in center_points2:
            tracking_objects[track_id] = pt
            kalman_filters[track_id] = create_kalman_filter()  # Create Kalman filter for new object
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
            kalman_filters[track_id] = create_kalman_filter()  # Create Kalman filter for new object
            track_id += 1

    count += 1
    for object_id, pt in tracking_objects.items():
        kalman = kalman_filters[object_id]
        prediction = kalman.predict()
        predicted_x, predicted_y = prediction[0], prediction[1]

        # Track and draw the predicted face
        cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (255, 255, 0), -1)
        cv2.putText(frame, str(object_id), (int(predicted_x), int(predicted_y) - 7), 0, 1, (0, 0, 255), 2)

    # Display the total number of tracked faces
    total_faces = len(tracking_objects)
    cv2.putText(frame, f"Total Faces: {total_faces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
