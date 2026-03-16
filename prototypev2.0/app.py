import webbrowser
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Decode the image from hte browser
    file =request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # ---Run your models here ---
    # bbox, landmarks, emotion, confidence = your_ensemble_model(frame)

    # Dummy response for now (replace with real model output)
    result = {
        "emotion": "happy",
        "confidence": 0.92,
        "bbox": [80, 60, 160, 160],             # [x, y, w, h] at capture resolution
        "landmarks": [[100,80],[120,80]]     # list of [x, y] points
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)