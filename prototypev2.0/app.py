import os
import json
import base64
import webbrowser
import threading
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
import mediapipe as mp
import dlib
import joblib
from PIL import Image
from scipy.special import softmax
from catboost import CatBoostClassifier
from flask import Flask, render_template, request, jsonify

# ─────────────── App Factory ───────────────
app = Flask(__name__)

# ─────────────── EmotionDetector ───────────────
class EmotionDetector:
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, model_dir="models"):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base_path, model_dir)

        # ── 1. Mediapipe ───────────────────────────
        self.face_mesh              = None
        self.mediapipe_available    = False
        self.mediapipe_init_error   = None

        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh    = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
                self.mediapipe_available = True
                print("✅ Mediapipe initialized.")
            except Exception as e:
                self.mediapipe_init_error = str(e)
                print(f"⚠️ Mediapipe failed to init: {e}")
        else:
            print("❌ mp.solutions not found — Mediapipe disabled.")

        # ── 2. Dlib ───────────────────────────
        dlib_fname = "shape_predictor_68_face_landmarks.dat"
        dlib_path = (
            os.path.join(self.model_dir, dlib_fname)
            if os.path.exists(os.path.join(self.model_dir, dlib_fname))
            else os.path.join(base_path, dlib_fname)
            if os.path.exists(os.path.join(base_path, dlib_fname))
            else None
        )

        self.detector       = dlib.get_frontal_face_detector()
        self.predictor      = None
        self.dlib_available = False

        if dlib_path:
            try:
                self.predictor      = dlib.shape_predictor(dlib_path)
                self.dlib_available = True
                print("✅ DLib shape predictor loaded.")
            except Exception as e:
                print(f"⚠️ Dlib predictor failed: {e}")
        else:
            print("⚠️ Dlib .dat file not found.")

        # ── 3. Load ML Models ───────────────────────────
        print(f"📂 Loading models from: {self.model_dir}")
        try:
            # A. ResNet-50
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'ℹ️ Using device: {self.device}')

            class ResNetFER(torch.nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.resnet = models.resnet50(weights=None)

                    self.features = torch.nn.Sequential(
                        *list(self.resnet.children())[:-1]
                    )

                    num_features = self.resnet.fc.in_features
                    self.resnet.fc = torch.nn.Sequential(
                        torch.nn.Dropout(0.3),              # resnet.fc.0
                        torch.nn.Linear(num_features, 256), # resnet.fc.1
                        torch.nn.ReLU(),                    # resnet.fc.2
                        torch.nn.Dropout(0.5),              # resnet.fc.3
                        torch.nn.Linear(256, num_classes)   # resnet.fc.4
                    )
                
                def forward(self, x):
                    return self.resnet(x)
                
            num_classes = len(self.EMOTIONS)
            self.cnn_model = ResNetFER(num_classes=num_classes)

            # Load saved weights into the architecture
            checkpoint = torch.load(
                os.path.join(self.model_dir, 'best_resnet_model.pth'),
                map_location=self.device
            )

            # Log checkpoint metadata at startup
            print(f"ℹ️ Checkpoint epoch:     {checkpoint.get('epoch', 'N/A')}")
            print(f"ℹ️ Best val accuracy:    {checkpoint.get('best_val_acc', 'N/A'):.2f}")
            print(f"ℹ️ Epochs trained:       {checkpoint.get('epochs_trained', 'N/A')}")

            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.to(self.device)
            self.cnn_model.eval() # critical — disables dropout and batchnorm training mode
            print("✅ ResNet-50 loaded.")

            # Transforms must exactly match what was used during training
            # Default: ImageNet normalization - update mean/std if your training differed
            self.cnn_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # B. CatBoost
            self.cat_model = CatBoostClassifier(verbose=0)
            self.cat_model.load_model(os.path.join(self.model_dir, 'catboost_fer.cbm'))
            print("✅ CatBoost loaded.")

            # C. Meta-classifier
            self.meta_model = joblib.load(
                os.path.join(self.model_dir, "meta_classifier.pkl")
            )
            print("✅ Meta-classifier loaded")
        
        except Exception as e:
            print(f"❌ Model loading failed {e}")
            raise   # stops startup immediately instead of running with no models loaded

        # ── 4. Temperature Scaling ───────────────────────────
        temp_path = os.path.join(self.model_dir, 'temperatures.json')
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temps = json.load(f)
            self.T_cnn      = float(temps.get("T_cnn",      1.0))
            self.T_catboost = float(temps.get("T_catboost", 1.0))
            print(f"✅ Temperature scaling loaded - "
                  f"T_cnn={self.T_cnn:.1f}, T_catboost={self.T_catboost:.1f}")
        else:
            self.T_cnn      = 1.0
            self.T_catboost = 1.0
            print("ℹ️ No temperatures.json found - "
                  "temperature scaling disabled (T=1.0 no-op).")
    
    # ── Helpers ───────────────────────────
    def _apply_temperature(self, probs: np.ndarray, T: float) -> np.ndarray:
        """
        Calibrate probabilities via temperature scaling.
        probs : (1, n_classes) numpy array
        T     : learned temperature scalar (1.0 = identity, no change)
        """
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        return softmax(logits / T, axis=1)
    
    def _predict_cnn(self, face_img_rgb: np.ndarray) -> np.ndarray:
        """
        Run PyTorch ResNet-50 inference on a cropped RGB face image.
        Returns a (1, n_classes) numpy probability array.
        """
        face_pil    = Image.fromarray(face_img_rgb)             # numpy → PIL
        face_tensor = self.cnn_transform(face_pil)              # apply transforms
        face_input  = face_tensor.unsqueeze(0).to(self.device)  # add batch dim → (1, 3, 224, 224)
        
        with torch.no_grad():                       # no gradients at inference
            logits = self.cnn_model(face_input)     # (1, n_classes)
            probs  = torch.softmax(logits, dim=1)   # normalize to probabilities
            return probs.cpu().numpy()              # → numpy (1, n_classes)

    def _extract_geometric_features(
            self, img_rgb: np.ndarray, face_rect: list, results_mp
    ) -> pd.DataFrame:
        """Build the feature row CatBoost expects (dlib 68 pts + MP 478 pts.)."""
        row = {}
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # ── Dlib 68 landmarks ───────────────────────────
        dlib_rect = dlib.rectangle(
            int(face_rect[0]),
            int(face_rect[1]),
            int(face_rect[0] + face_rect[2]),
            int(face_rect[1] + face_rect[3]),
        )
        if self.dlib_available and self.predictor:
            try:
                shape = self.predictor(gray, dlib_rect)
                for k in range(68):
                    row[f'dlib_x{k}'] = shape.part(k).x
                    row[f'dlib_y{k}'] = shape.part(k).y
            except Exception:
                for k in range(68):
                    row[f'dlib_x{k}'] = 0
                    row[f'dlib_y{k}'] = 0
        else:
            for k in range(68):
                row[f'dlib_x{k}'] = 0
                row[f'dlib_y{k}'] = 0
        
        # ── MediaPipe 478 landmarks ───────────────────────────
        if results_mp and results_mp.multi_face_landmarks:
            landmarks = results_mp.multi_face_landmarks[0].landmark
            for k, lm in enumerate(landmarks):
                row[f'mp_x{k}'] = lm.x
                row[f'mp_y{k}'] = lm.y
                row[f'mp_z{k}'] = lm.z
        else:
            for k in range(478):
                row[f'mp_x{k}'] = 0
                row[f'mp_y{k}'] = 0
                row[f'mp_z{k}'] = 0
        
        return pd.DataFrame([row])
    
    # ── Main inference entry-point ───────────────────────────
    def process_frame(self, base64_image: str, run_comparison: bool = False) -> dict:
        """
        Accepts a base64-encoded image string (data-URL or raw base 64).
        Returns a dict with emotion label, confidence, bbox, and optional
        per-model breakdown when run_comparison=True.
        """
        NO_FACE = {"label": "No Face", "confidence": 0.0,
                   "bbox": [], "landmarks": []}
        
        try:
            # ── Decode Image ───────────────────────────
            raw      = base64_image.split(",")[1] if "," in base64_image else base64_image
            img_data = np.frombuffer(base64.b64decode(raw), np.uint8)
            img      = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            if img is None:
                return {"label": "Error", "confidence": 0.0,
                        "bbox": [], "landmarks": []}
            
            img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _    = img.shape
            results_mp = None
            bbox       = []

            # ── Face detection ───────────────────────────
            if self.face_mesh:
                results_mp = self.face_mesh.process(img_rgb)
                if results_mp.multi_face_landmarks:
                    lms     = results_mp.multi_face_landmarks[0].landmark
                    x_vals  = [lm.x * w for lm in lms]
                    y_vals  = [lm.y * h for lm in lms]
                    x_min   = max(0, min(x_vals))
                    x_max   = min(w, max(x_vals))
                    y_min   = max(0, min(y_vals))
                    y_max   = min(h, max(y_vals))
                    bbox    = [int(x_min), int(y_min),
                               int(x_max - x_min), int(y_max - y_min)]
            else:
                faces = self.detector(img_rgb, 0)
                if not faces:
                    return NO_FACE
            
            # ── Crop face ───────────────────────────
            x, y, wb, hb = bbox
            face_img     = img_rgb[y:y + hb, x:x + wb]
            if face_img.size == 0:
                return NO_FACE
            
            # ── Extract landmarks for frontend drawing ───────────────────────────
            landmarks_out = []
            if results_mp and results_mp.multi_face_landmarks:
                lms = results_mp.multi_face_landmarks[0].landmark
                landmarks_out = [
                    [int(lm.x * w), int(lm.y * h)]
                    for lm in lms
                ]
            
            # ── Geometric features for CatBoost ───────────────────────────
            geo_features = self._extract_geometric_features(
                img_rgb, bbox, results_mp
            )

            # ── Base model predictions ───────────────────────────
            cnn_probs_raw = self._predict_cnn(face_img)                 # (1, 6) PyTorch
            cat_probs_raw = self.cat_model.predict_proba(geo_features)  # (1, 6)

            # ── Temperature scaling (no-op when T=1.0) ───────────────────────────
            cnn_probs_cal = self._apply_temperature(cnn_probs_raw, self.T_cnn)
            cat_probs_cal = self._apply_temperature(cat_probs_raw, self.T_catboost)

            # ── CNN-only result (for comparison mode) ───────────────────────────
            cnn_idx    = int(np.argmax(cnn_probs_cal[0]))
            cnn_result = {
                "label":    self.EMOTIONS[cnn_idx],
                "confidence": float(cnn_probs_cal[0][cnn_idx]),
            }

            # ── Ensemble via meta-classifier ───────────────────────────
            stacked     = np.hstack((cnn_probs_cal, cat_probs_cal)) # (1, 12)
            final_dist  = self.meta_model.predict_proba(stacked)[0] # (6,)
            ensemble_idx = int(np.argmax(final_dist))

            ensemble_result = {
                "label":      self.EMOTIONS[ensemble_idx],
                "confidence": float(final_dist[ensemble_idx]),
                "bbox":       [int(v) for v in bbox],
                "landmarks":  landmarks_out,
            }

            # ── Return ───────────────────────────
            if run_comparison:
                return {
                    "ensemble":  ensemble_result,
                    "cnn_only":  cnn_result,
                    "bbox":      [int(v) for v in bbox],
                    "landmarks": landmarks_out
                }
            
            return ensemble_result
        except Exception as e:
            print(f"❌ process_frame error: {e}")
            return {"label": "Error", "confidence": 0.0,
                    "bbox": [], "landmarks": []}
        
# ─────────────── Initialize detector (once at startup) ───────────────            
detector = EmotionDetector(model_dir="models")

# ─────────────── Routes ───────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body:
        { "frame": "<based64-encoded image>", "compare": false }

    Returns JSON:
        {
            "label":      "Happy",
            "confidence": 0.91,
            "bbox":       [x, y, w, h]
            "landmarks":  []
        }
    """

    data = request.get_json(force=True, silent=True)

    if not data or 'frame' not in data:
        return jsonify({"error": "Missing 'frame' key in JSON body." }), 400
    
    run_comparison = bool(data.get('compare', False))
    result         = detector.process_frame(data['frame'], run_comparison)

    return jsonify(result)

    # # Decode the image from hte browser
    # file =request.files['frame']
    # npimg = np.frombuffer(file.read(), np.uint8)
    # frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # # ---Run your models here ---
    # # bbox, landmarks, emotion, confidence = your_ensemble_model(frame)

    # # Dummy response for now (replace with real model output)
    # result = {
    #     "emotion": "Happy",
    #     "confidence": 0.92,
    #     "bbox": [80, 60, 160, 160],             # [x, y, w, h] at capture resolution
    #     "landmarks": [[100,80],[120,80]]     # list of [x, y] points
    # }
    # return jsonify(result)

# ─────────────── Entry point ───────────────
if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, host='0.0.0.0', port=5000)