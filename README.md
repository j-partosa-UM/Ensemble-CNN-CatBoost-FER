# Ensemble CNN–CatBoost Facial Expression Recognition

A thesis prototype for real-time facial expression recognition using an ensemble of ResNet-50 (CNN) and CatBoost, with facial landmark extraction via MediaPipe and Dlib.

## Overview
- **CNN (ResNet-50):** Processes face crops at 224×224 for visual feature extraction
- **CatBoost:** Uses 1,610 geometric landmark features (Dlib + MediaPipe)
- **Meta-Learner (Logistic Regression):** Combines both models' probability outputs for final prediction
- **Live Inference:** Real-time webcam feed with per-frame emotion classification

## Requirements
- **Python 3.11 is required**
- If you have a higher Python version, use Conda to create a virtual environment:
```bash
conda create -n fer_env python=3.11
conda activate fer_env
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/fsayson-sketch/prototypev2.git
cd prototypev2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

## Notes
- GPU is used automatically if available (CUDA), otherwise falls back to CPU
- This is a prototype — performance may vary depending on hardware

## Tech Stack
- **Language:** Python 3.11, HTML, CSS, JavaScript
- **Frameworks:** Flask, PyTorch, CatBoost, MediaPipe, Dlib
- **Environment:** pip / Conda

## License
This project is open-source and available under the [MIT License](LICENSE).