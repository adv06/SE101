import cv2
import numpy as np
from object_detection import ObjectDetection
from math import sqrt

# Object detection using YOLO 
# Initalize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture(0)

#get frame number
# get all frames, video is a bunch of frames
count = 0 

tracking_objects = {}
track_id = 0
while True:
    # ret - true/false, frame - frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    if not ret:
        break 
    # Detect objkects on frame
    # class id - what type ofo vehicles, score - confidence, boxes - bounding box of location
    (class_ids, scores, boxes) = od.detect(frame)
    
    center_points2 = []

    for box in boxes:
        (x, y, w, h) = box
        cx = (x+x+w) // 2
        cy = (y+y+h) // 2
        center_points2.append((cx, cy))
        # print(f"Frame {count} Rectange {x, y, w, h}")
        
        # make rectangles on frame for objects
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 200), 2)
    
    #only at begininning 
    if count == 0:
        for pt in center_points2:
            tracking_objects[track_id] = pt
            track_id += 1 
        
    else:
        # check previous objects
        for object_id, pt in tracking_objects.copy().items():
            object_exists = False
            for pt2 in center_points2.copy():
                euc = sqrt((pt[0]-pt2[0])**2+(pt[1]-pt2[1])**2)
                # update object position
                if euc < 20:
                    tracking_objects[object_id] = pt2
                    object_exists = True 
                    center_points2.remove(pt2)
                    break 
            # remove id if exists
            if not object_exists:
                tracking_objects.pop(object_id)
    if len(center_points2) != 0 and count != 0:
        for pt in center_points2:
            tracking_objects[track_id] = pt
            track_id += 1 
    count += 1
    for object_id, pt in tracking_objects.items():
        # create circles at the centroid
        cv2.circle(frame, pt, 5, (255, 255, 0), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1]-7), 0, 1, (0, 0, 255), 2)

    # print("Tracking objects")

    cv2.imshow("Frame", frame)
    # key = cv2.waitKey(0)

    # if key == 27:
    #     break 

#release and destroy video frames
cap.release()
cv2.destroyAllWindows()

