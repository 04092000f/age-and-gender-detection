import cv2
import numpy as np

class AgeGenderDetector:
    def __init__(self, model_paths):
        self.faceNet = cv2.dnn.readNet(model_paths["face_model"], model_paths["face_config"])
        self.ageNet = cv2.dnn.readNet(model_paths["age_model"], model_paths["age_config"])
        self.genderNet = cv2.dnn.readNet(model_paths["gender_model"], model_paths["gender_config"])

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-30)',
                        '(31-40)', '(41-50)', '(51-70)', '70+']
        self.genderList = ['Male', 'Female']

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True)
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype(int))

        return faces

    def predict(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

        self.genderNet.setInput(blob)
        gender = self.genderList[self.genderNet.forward()[0].argmax()]

        self.ageNet.setInput(blob)
        age = self.ageList[self.ageNet.forward()[0].argmax()]

        return gender, age