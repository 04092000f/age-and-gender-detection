import cv2

class VideoStream:
    def __init__(self, detector):
        self.detector = detector

    def generate(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detector.detect_faces(frame)

            for (x1, y1, x2, y2) in faces:
                face_crop = frame[y1:y2, x1:x2]
                gender, age = self.detector.predict(face_crop)
                label = f"{gender}, {age}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')