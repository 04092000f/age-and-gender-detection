from flask import Flask, Response, render_template
from utils.detector import AgeGenderDetector
from utils.streamer import VideoStream

# Flask App
app = Flask(__name__)

model_paths = {
    "face_model": "models/opencv_face_detector_uint8.pb",
    "face_config": "models/opencv_face_detector.pbtxt",
    "age_model": "models/age_net.caffemodel",
    "age_config": "models/age_deploy.prototxt",
    "gender_model": "models/gender_net.caffemodel",
    "gender_config": "models/gender_deploy.prototxt"
}

detector = AgeGenderDetector(model_paths)
streamer = VideoStream(detector)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/video')
def video_feed():
    return Response(streamer.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)