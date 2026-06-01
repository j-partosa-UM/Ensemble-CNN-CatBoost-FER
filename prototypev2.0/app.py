import os
import json
import base64
import webbrowser
import threading
import sys
import urllib.request

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
        self.face_landmarker        = None
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
            # mediapipe>=0.10.33 may ship Tasks API without exposing legacy mp.solutions
            model_path = os.path.join(self.model_dir, "face_landmarker.task")
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            try:
                if not os.path.exists(model_path):
                    print("ℹ️ Downloading FaceLandmarker task model...")
                    urllib.request.urlretrieve(model_url, model_path)

                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python import vision

                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
                self.mediapipe_available = True
                print("✅ Mediapipe Tasks FaceLandmarker initialized.")
            except Exception as e:
                self.mediapipe_init_error = str(e)
                print(f"⚠️ Mediapipe Tasks init failed: {e}")
                print("ℹ️ Falling back to Dlib-only face detection.")

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
            print("\nDownloading dlib's 68-point predictor...")

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

                    num_features = self.resnet.fc.in_features
                    self.resnet.fc = torch.nn.Sequential(
                        torch.nn.Linear(num_features, 512),
                        torch.nn.BatchNorm1d(512),
                        torch.nn.Dropout(0.5),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(512, num_classes)
                    )
                
                def forward(self, x):
                    return self.resnet(x)
                
            num_classes = len(self.EMOTIONS)
            self.cnn_model = ResNetFER(num_classes=num_classes)

            # Load saved weights into the architecture
            checkpoint = torch.load(
                os.path.join(self.model_dir, 'best_resnet_model.pth'),
                map_location=self.device,
                weights_only=False
            )

            # Log checkpoint metadata at startup
            print(f"ℹ️ Checkpoint epoch:     {checkpoint.get('epoch', 'N/A')}")
            print(f"ℹ️ train accuracy:    {checkpoint.get('train_acc', 'N/A'):.2f}")
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
            self.cat_model.load_model(os.path.join(self.model_dir, 'catboost_landmarks_model.cbm'))
            print("✅ CatBoost loaded.")

            # C. Meta-classifier
            self.meta_model = joblib.load(
                os.path.join(self.model_dir, "meta_learner.pkl")
            )
            print("✅ Meta-classifier loaded")
        
        except Exception as e:
            print(f"❌ Model loading failed {e}")
            raise   # stops startup immediately instead of running with no models loaded

        # ── 4. Temperature Scaling ───────────────────────────
        temp_path = os.path.join(self.model_dir, 'optimal_temperature.json')
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temps = json.load(f)
            self.T_cnn      = float(temps.get("T_cnn",      1.0))
            self.T_catboost = float(temps.get("T_catboost", 1.0))
            print(f"✅ Temperature scaling loaded - "
                  f"T_cnn={self.T_cnn:.4f}, T_catboost={self.T_catboost:.4f}")
        else:
            self.T_cnn      = 1.0
            self.T_catboost = 1.0
            print("ℹ️ No optimal_temperature.json found - "
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

    def _extract_mp_landmarks(self, results_mp):
        """
        Return normalized landmarks for either API shape:
        - Legacy: results.multi_face_landmarks[0].landmark
        - Tasks:  results.face_landmarks[0]
        """
        if not results_mp:
            return []

        if hasattr(results_mp, 'multi_face_landmarks') and results_mp.multi_face_landmarks:
            return results_mp.multi_face_landmarks[0].landmark

        if hasattr(results_mp, 'face_landmarks') and results_mp.face_landmarks:
            return results_mp.face_landmarks[0]

        return []

    def _extract_geometric_features(
            self, img_rgb: np.ndarray, face_rect: list, results_mp
    ) -> pd.DataFrame:
        """
        Build the 1610-feature row CatBoost expects.
        Layout: dlib(136) + MP(1434) + MP_dist(20) + dlib_dist(20) = 1610.
        Feature order mirrors the training feature-engineering cell.
        """
        h_img, w_img = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        DLIB_FEAT_DIM = 136
        MP_FEAT_DIM = 1434   # 478 * 3
        N_MP_DIST = 20
        N_DLIB_DIST = 20

        # ── dlib 68 landmarks (normalized) ──────────────────────────────────────
        dlib_rect = dlib.rectangle(
            int(face_rect[0]), int(face_rect[1]),
            int(face_rect[0] + face_rect[2]),
            int(face_rect[1] + face_rect[3]),
        )
        dlib_xy = np.zeros((68, 2), dtype=np.float32)
        if self.dlib_available and self.predictor:
            try:
                shape = self.predictor(gray, dlib_rect)
                for k in range(68):
                    dlib_xy[k, 0] = shape.part(k).x / max(w_img, 1)
                    dlib_xy[k, 1] = shape.part(k).y / max(h_img, 1)
            except Exception:
                pass

        # Flatten dlib first to preserve training order.
        dlib_raw = dlib_xy.reshape(-1)

        # ── MP 478 landmarks (x, y, z) ─────────────────────────────────────────
        landmarks = self._extract_mp_landmarks(results_mp)
        n_mp = 478
        mp_xyz = np.zeros((n_mp, 3), dtype=np.float32)
        if landmarks:
            use_n = min(len(landmarks), n_mp)
            for k in range(use_n):
                mp_xyz[k, 0] = float(landmarks[k].x)
                mp_xyz[k, 1] = float(landmarks[k].y)
                mp_xyz[k, 2] = float(getattr(landmarks[k], 'z', 0.0))

        # Flatten MP second to preserve training order.
        mp_raw = mp_xyz.reshape(-1)

        # MP landmark indices — mirrors training code exactly
        MP_IDX = {
            'left_eye_outer':   33,
            'left_eye_inner':   133,
            'left_eye_top':     159,
            'left_eye_bottom':  145,
            'right_eye_inner':  362,
            'right_eye_outer':  263,
            'right_eye_top':    386,
            'right_eye_bottom': 374,
            'left_brow_peak':   70,
            'right_brow_peak':  299,
            'mouth_left':       61,
            'mouth_right':      291,
            'mouth_top':        0,
            'mouth_bottom':     17,
            'nose_tip':         1,
            'chin':             152,
            'jaw_left':         234,
            'jaw_right':        454,
            'lower_lip_bottom': 16,
            'left_eye_center':  468,
            'right_eye_center': 473,
            'upper_lip_top':    267,
        }


        # ── 20 MP-derived distance features — mirrors training ─────────────────
        def pt(key):
            idx = MP_IDX[key]
            return np.array([mp_xyz[idx, 0], mp_xyz[idx, 1]], dtype=np.float32)

        mp_dist = np.zeros((N_MP_DIST,), dtype=np.float32)
        if not np.all(mp_raw == 0):
            left_eye_w   = np.linalg.norm(pt('left_eye_outer')  - pt('left_eye_inner'))
            right_eye_w  = np.linalg.norm(pt('right_eye_inner') - pt('right_eye_outer'))
            mouth_w      = np.linalg.norm(pt('mouth_left')      - pt('mouth_right'))
            mouth_h      = np.linalg.norm(pt('mouth_top')       - pt('mouth_bottom'))
            left_brow_h  = np.linalg.norm(pt('left_brow_peak')  - pt('left_eye_top'))
            right_brow_h = np.linalg.norm(pt('right_brow_peak') - pt('right_eye_top'))
            left_eye_h   = np.linalg.norm(pt('left_eye_top')    - pt('left_eye_bottom'))
            right_eye_h  = np.linalg.norm(pt('right_eye_top')   - pt('right_eye_bottom'))
            nose_mouth   = np.linalg.norm(pt('nose_tip')        - pt('mouth_top'))
            face_h       = np.linalg.norm(pt('chin')            - pt('nose_tip'))

            corner_l_y         = float(pt('mouth_left')[1])
            corner_r_y         = float(pt('mouth_right')[1])
            lip_center_y       = float(pt('mouth_top')[1])
            mouth_corner_curve = ((corner_l_y + corner_r_y) / 2 - lip_center_y) / (face_h + 1e-6)

            lid_exposure = ((left_eye_h + right_eye_h) / 2) / (face_h + 1e-6)
            chin_y = float(pt('chin')[1])
            lower_lip_y = float(pt('lower_lip_bottom')[1])
            chin_to_lower_lip = abs(chin_y - lower_lip_y) / (face_h + 1e-6)

            left_brow_to_eye = np.linalg.norm(pt('left_brow_peak') - pt('left_eye_center'))
            right_brow_to_eye = np.linalg.norm(pt('right_brow_peak') - pt('right_eye_center'))
            brow_to_eye_center = ((left_brow_to_eye + right_brow_to_eye) / 2) / (face_h + 1e-6)

            upper_lip_raise = np.linalg.norm(pt('nose_tip') - pt('upper_lip_top')) / (face_h + 1e-6)

            mp_dist[:] = [
                left_eye_w,                                      # 0
                right_eye_w,                                     # 1
                mouth_w,                                         # 2
                mouth_h,                                         # 3
                left_brow_h,                                     # 4
                right_brow_h,                                    # 5
                left_eye_h,                                      # 6
                right_eye_h,                                     # 7
                mouth_w    / (mouth_h      + 1e-6),              # 8  mouth_aspect_ratio
                left_eye_h / (left_eye_w   + 1e-6),              # 9  eye_aspect_ratio
                mouth_h    / (face_h       + 1e-6),              # 10 mouth_openness
                mouth_w    / (face_h       + 1e-6),              # 11 lip_tightness
                left_brow_h  / (face_h     + 1e-6),              # 12 brow_raise_left
                right_brow_h / (face_h     + 1e-6),              # 13 brow_raise_right
                min(nose_mouth / (mouth_h  + 1e-6), 20.0),       # 14 nose_mouth_ratio
                mouth_corner_curve,                              # 15
                lid_exposure,                                    # 16
                chin_to_lower_lip,                               # 17
                brow_to_eye_center,                              # 18
                upper_lip_raise,                                 # 19
            ]

        # ── 20 dlib-derived distance features — mirrors training ───────────────
        dlib_dist = np.zeros((N_DLIB_DIST,), dtype=np.float32)
        if not np.all(dlib_raw == 0):
            d = dlib_xy

            left_eye_w   = np.linalg.norm(d[36] - d[39])
            right_eye_w  = np.linalg.norm(d[42] - d[45])
            mouth_w      = np.linalg.norm(d[48] - d[54])
            mouth_h      = np.linalg.norm(d[51] - d[57])
            left_brow_h  = np.linalg.norm(d[19] - d[37])
            right_brow_h = np.linalg.norm(d[24] - d[44])
            left_eye_h   = np.linalg.norm(d[37] - d[41])
            right_eye_h  = np.linalg.norm(d[43] - d[47])
            mouth_nose   = np.linalg.norm(d[33] - d[51])
            jaw_w        = np.linalg.norm(d[0]  - d[16])
            face_h       = np.linalg.norm(d[8]  - d[27])

            mouth_corner_curve = ((d[48][1] + d[54][1]) / 2 - d[51][1]) / (face_h + 1e-6)
            lid_exposure = ((left_eye_h + right_eye_h) / 2) / (face_h + 1e-6)
            chin_to_lower_lip = abs(d[8][1] - d[57][1]) / (face_h + 1e-6)

            left_eye_center = (d[37] + d[41]) / 2
            right_eye_center = (d[43] + d[47]) / 2
            left_brow_to_eye = np.linalg.norm(d[19] - left_eye_center)
            right_brow_to_eye = np.linalg.norm(d[24] - right_eye_center)
            brow_to_eye_center = ((left_brow_to_eye + right_brow_to_eye) / 2) / (face_h + 1e-6)

            upper_lip_raise = np.linalg.norm(d[33] - d[51]) / (face_h + 1e-6)

            dlib_dist[:] = [
                left_eye_w,
                right_eye_w,
                mouth_w,
                mouth_h,
                left_brow_h,
                right_brow_h,
                left_eye_h,
                right_eye_h,
                mouth_nose,
                mouth_w    / (mouth_h + 1e-6),
                left_eye_h / (left_eye_w + 1e-6),
                jaw_w,
                face_h,
                mouth_h / (face_h + 1e-6),
                min(mouth_nose / (mouth_h + 1e-6), 20.0),
                mouth_corner_curve,
                lid_exposure,
                chin_to_lower_lip,
                brow_to_eye_center,
                upper_lip_raise,
            ]

        # Final order must match training: dlib_raw | mp_raw | mp_dist | dlib_dist
        feature_row = np.hstack([dlib_raw, mp_raw, mp_dist, dlib_dist]).astype(np.float32)
        if feature_row.shape[0] != (DLIB_FEAT_DIM + MP_FEAT_DIM + N_MP_DIST + N_DLIB_DIST):
            raise ValueError(f"Unexpected feature length: {feature_row.shape[0]} (expected 1610)")

        return pd.DataFrame([feature_row])
    
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
                lms = self._extract_mp_landmarks(results_mp)
                if lms:
                    x_vals  = [lm.x * w for lm in lms]
                    y_vals  = [lm.y * h for lm in lms]
                    x_min   = max(0, min(x_vals))
                    x_max   = min(w, max(x_vals))
                    y_min   = max(0, min(y_vals))
                    y_max   = min(h, max(y_vals))
                    bbox    = [int(x_min), int(y_min),
                               int(x_max - x_min), int(y_max - y_min)]
            elif self.face_landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                results_mp = self.face_landmarker.detect(mp_image)
                lms = self._extract_mp_landmarks(results_mp)
                if lms:
                    x_vals = [lm.x * w for lm in lms]
                    y_vals = [lm.y * h for lm in lms]
                    x_min = max(0, min(x_vals))
                    x_max = min(w, max(x_vals))
                    y_min = max(0, min(y_vals))
                    y_max = min(h, max(y_vals))
                    bbox = [int(x_min), int(y_min),
                            int(x_max - x_min), int(y_max - y_min)]
            else:
                faces = self.detector(img_rgb, 0)
                if not faces:
                    return NO_FACE
                face = faces[0]
                x1 = max(0, int(face.left()))
                y1 = max(0, int(face.top()))
                x2 = min(w, int(face.right()))
                y2 = min(h, int(face.bottom()))
                bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]

            if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                return NO_FACE
            
            # ── Crop face ───────────────────────────
            x, y, wb, hb = bbox
            face_img     = img_rgb[y:y + hb, x:x + wb]
            if face_img.size == 0:
                return NO_FACE
            
            # ── Extract landmarks for frontend drawing ───────────────────────────
            landmarks_out = []
            lms = self._extract_mp_landmarks(results_mp)
            if lms:
                landmarks_out = [
                    [int(lm.x * w), int(lm.y * h)]
                    for lm in lms
                ]
            
            # ── Geometric features for CatBoost ───────────────────────────
            geo_features = self._extract_geometric_features(
                img_rgb, bbox, results_mp
            )

            # ── Base model predictions ───────────────────────────
            cnn_probs_raw = self._predict_cnn(face_img)                 # (1, 7) PyTorch
            cat_probs_raw = self.cat_model.predict_proba(geo_features)  # (1, 7)

            
            # ── Temperature scaling (no-op when T=1.0) ───────────────────────────
            cnn_probs_cal = self._apply_temperature(cnn_probs_raw, self.T_cnn)
            cat_probs_cal = self._apply_temperature(cat_probs_raw, self.T_catboost)

            # ── CNN-only result (for comparison mode) ───────────────────────────
            cnn_idx    = int(np.argmax(cnn_probs_cal[0]))
            cnn_result = {
                "label":      self.EMOTIONS[cnn_idx],
                "confidence": float(cnn_probs_cal[0][cnn_idx]),
                "probs":      {e: float(p) for e, p in zip(self.EMOTIONS, cnn_probs_cal[0])},
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
                "probs":      {e: float(p) for e, p in zip(self.EMOTIONS, final_dist)},
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
    from livereload import Server
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5500")).start()

    server = Server(app.wsgi_app)

    server.serve(host='0.0.0.0', port=5500, debug=False)
    # app.run(debug=False, host='0.0.0.0', port=5000)