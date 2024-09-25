import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

class Simple_face:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.face_names = ['Rak Sa', 'Sedtha']

    def detect_known_faces(self, frame):
        face_locations = self.detect_faces(frame)
        results = []
        for (y1, x1, y2, x2) in face_locations:
            face_roi = frame[y1:y2, x1:x2]
            processed_face = self.preprocess_frame(face_roi)
            predictions = self.model.predict(processed_face)
            face_name = self.process_predictions(predictions)
            results.append((y1, x1, y2, x2, face_name))
        return [(y1, x1, y2, x2) for y1, x1, y2, x2, _ in results], [face_name for _, _, _, _, face_name in results]

    def preprocess_frame(self, frame):
        resized_frame = cv2.resize(frame, (300, 300))
        normalized_frame = resized_frame / 255.0
        return np.expand_dims(normalized_frame, axis=0)

    def process_predictions(self, predictions):
        class_index = np.argmax(predictions[0])
        return self.face_names[class_index]

    def detect_faces(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.10, minNeighbors=5, minSize=(30,30))
        return [(y, x, y + h, x + w) for (x, y, w, h) in faces]

model_path = ("C:/Users/sedth/PycharmProjects/pythonProject/Face Recognition/model1.h5")
sfr = Simple_face(model_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1-30), (x2, y1), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, name, (x2, y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow("Fram ", frame)

    key = cv2.waitKey(1)
    if key == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
