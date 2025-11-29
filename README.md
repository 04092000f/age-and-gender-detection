# Real-Time Age & Gender Detection Web App

This project is a real-time Age & Gender Detection system built using OpenCV, Flask, and Deep Learning models. It captures live webcam video, detects faces, predicts gender and age range, and streams the processed video to a browser.

---

## Table of Contents

- [Demo Preview](#demo-preview)
- [Technologies Used](#technologies-used)
- [Models Used](#models-used)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [Contributions](#contributions)
- [Feedback](#feedback)

---
## Demo Preview

Live prediction happens inside the browser using MJPEG streaming.

| Feature | Status |
|--------|--------|
| Real-time Face Detection | Yes |
| Age Prediction | Yes |
| Gender Prediction | Yes |
| Web Streaming (Flask) | Yes |
| Webcam Support | Yes |

---

## Technologies Used

- Python  
- Flask  
- OpenCV (DNN Module)  
- Pre-trained Caffe Models  
- Webcam Streaming  

---

## Models Used

| Model Type | Files |
|-----------|-------|
| Face Detection | `opencv_face_detector.pbtxt`, `opencv_face_detector_uint8.pb` |
| Age Detection | `age_deploy.prototxt`, `age_net.caffemodel` |
| Gender Detection | `gender_deploy.prototxt`, `gender_net.caffemodel` |

Make sure these files are placed inside the `models/` folder.

---


## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


### 3. Download the Models file

- Click [here](https://drive.google.com/drive/folders/129pGjmnPmbWs6exgGfP7A4pYp9YjKQBe?usp=drive_link) To download all the models required for running the app.
- Once downloaded, store all the required files in `models` directory.

### 4. Run the Application

```bash
python app.py
```

### 5. Access in Browser

```
http://127.0.0.1:5000 or localhost:5000
```


## Project Structure
```
📂 age-and-gender-detection/
│── app.py
│── requirements.txt
│── templates/
│   └── index.html
│── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
└── README.md
```

## How It Works

1. Flask initializes a web server.

2. The webcam stream is processed frame-by-frame using OpenCV.

3. The face detector model locates faces in each frame.

4. Each detected face is passed to the age and gender deep learning models.

5. Predictions are drawn on the video and streamed to the browser using MJPEG streaming.


## Future Improvements

- Deployment on cloud (Render, AWS, or Railway)

- Convert models to ONNX for faster inference

- Improved Web UI

- Add emotion recognition or face recognition


## Contributions

- Contributions and suggestions are welcome.

- If you like this project, give it a star on GitHub.


## Feedback

- If you have ideas to improve this project, feel free to open an issue or connect via [LinkedIn](https://www.linkedin.com/in/furqansa344/).
