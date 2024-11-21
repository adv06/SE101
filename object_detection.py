import cv2
import numpy as np

class FaceDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
        print("Loading Face Detection Model with YOLOv4")
        self.nmsThreshold = 0.4  # Adjusted for better detection
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = ["face"]  # Limit to face class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def detect(self, frame):
        # Use only face class (index 0 in YOLOv4 model)
        class_ids, confidences, boxes = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        
        faces = []
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            if class_id == 0:  # Class ID for "face" in the trained model
                x, y, w, h = box
                faces.append((x, y, w, h))  # Collect face bounding box coordinates

        return faces
