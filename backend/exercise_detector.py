from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add parent directory to path to import exercise detection modules
# This assumes the exercise detection scripts are in the parent directory
backend_dir = Path(__file__).parent.parent.parent
parent_dir = backend_dir.parent
sys.path.insert(0, str(parent_dir))


# ─────────────────────────────────────────────────────────────────────────────
# GCN model definition — must match the training notebook exactly
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Landmark order taken directly from bicep_curl.py → init_important_landmarks()
_BICEP_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "right_elbow",    "left_elbow",      # RIGHT before LEFT — matches source
    "right_wrist",    "left_wrist",      # RIGHT before LEFT — matches source
    "left_hip",       "right_hip",
]
_N_NODES = len(_BICEP_LANDMARKS)   # 9
_N_FEATS = 4                       # x, y, z, visibility

_SKELETON_EDGES = [
    ("nose",          "left_shoulder"),
    ("nose",          "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder","right_elbow"),
    ("left_elbow",    "left_wrist"),
    ("right_elbow",   "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder","right_hip"),
    ("left_hip",      "right_hip"),
]

_node_to_idx = {n: i for i, n in enumerate(_BICEP_LANDMARKS)}
_src, _dst = [], []
for _u, _v in _SKELETON_EDGES:
    _i, _j = _node_to_idx[_u], _node_to_idx[_v]
    _src += [_i, _j]
    _dst += [_j, _i]
_EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)

# Column order that matches the scaler fitted during training
_BICEP_FEATURE_COLS = [
    f"{lm}_{coord}"
    for lm in _BICEP_LANDMARKS
    for coord in ["x", "y", "z", "v"]
]


class _PoseGCN(nn.Module):
    """
    3-layer GCN with flatten (not global_mean_pool).
    Architecture must be identical to the training notebook.
    """
    def __init__(self, in_feats=4, hidden=64, out_feats=32,
                 n_classes=2, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats

        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(out_feats)

        # 9 nodes × 32 features = 288
        self.head = nn.Sequential(
            nn.Linear(_N_NODES * out_feats, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        batch_size = batch.max().item() + 1
        x = x.view(batch_size, _N_NODES * self.out_feats)
        return self.head(x)


def _normalize_pose_row(row: dict) -> dict:
    """
    Apply camera-invariant normalisation used during GCN training:
      1. Subtract torso centre  → removes camera position offset
      2. Divide by torso size   → removes camera distance / body height
    """
    centre_x = (row["left_shoulder_x"] + row["right_shoulder_x"] +
                 row["left_hip_x"]      + row["right_hip_x"]) / 4
    centre_y = (row["left_shoulder_y"] + row["right_shoulder_y"] +
                 row["left_hip_y"]      + row["right_hip_y"]) / 4

    sh_mid_x = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sh_mid_y = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hi_mid_x = (row["left_hip_x"]      + row["right_hip_x"])      / 2
    hi_mid_y = (row["left_hip_y"]      + row["right_hip_y"])      / 2

    torso_size = ((sh_mid_x - hi_mid_x) ** 2 +
                  (sh_mid_y - hi_mid_y) ** 2) ** 0.5 + 1e-6

    out = dict(row)
    for lm in _BICEP_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - centre_x) / torso_size
        out[f"{lm}_y"] = (row[f"{lm}_y"] - centre_y) / torso_size
        out[f"{lm}_z"] =  row[f"{lm}_z"]              / torso_size
        # visibility unchanged
    return out


def _predict_posture_gcn(
    raw_row: dict,
    gcn_model: _PoseGCN,
    scaler,
    device: torch.device,
    threshold: float = 0.95,
) -> tuple:
    """
    Run a single-frame prediction through the GCN.

    Returns
    -------
    predicted_class : str   "C" or "L"
    confidence      : float max probability
    """
    import numpy as np

    # 1. Normalise
    norm = _normalize_pose_row(raw_row)

    # 2. Build feature vector in the same column order the scaler was fitted on
    feat_vec = [[norm[col] for col in _BICEP_FEATURE_COLS]]

    # 3. Scale
    feat_scaled = scaler.transform(feat_vec)

    # 4. Build single-graph Data object
    x_node = torch.tensor(
        feat_scaled[0].reshape(_N_NODES, _N_FEATS), dtype=torch.float
    )
    data = Data(x=x_node, edge_index=_EDGE_INDEX)
    data.batch = torch.zeros(_N_NODES, dtype=torch.long)  # single graph
    data = data.to(device)

    # 5. Inference
    with torch.no_grad():
        logits = gcn_model(data)              # [1, 2]
        probs  = F.softmax(logits, dim=1)[0]  # [2]

    prob_c = probs[0].item()
    prob_l = probs[1].item()
    confidence = max(prob_c, prob_l)
    predicted_class = "L" if prob_l >= (1 - threshold) else "C"

    return predicted_class, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ExerciseDetector(ABC):
    """Abstract base class for exercise detectors"""

    @abstractmethod
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process video and return exercise results.

        Args:
            video_path: Path to input video file
            output_path: Optional path to save processed video (for visualization)

        Returns:
            Dictionary with keys:
            - score: int (0-100 quality score or rep count)
            - duration_seconds: float
            - error_count: int
            - details: dict (exercise-specific details)
        """
        pass


# ─────────────────────────────────────────────────────────────────────────────
# BicepCurlDetector  — uses the new GCN model for lean-back detection
#                      all arm analysis logic is unchanged
# ─────────────────────────────────────────────────────────────────────────────

class BicepCurlDetector(ExerciseDetector):
    """Bicep curl exercise detector — GCN-based lean-back classification"""

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process bicep curl video"""
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
        import time
        import pickle

        # ── Load GCN model and scaler ─────────────────────────────────────
        gcn_weights_file = parent_dir / "bicep_curl_gcn.pth"
        gcn_scaler_file  = parent_dir / "bicep_curl_gcn_scaler.pkl"

        if not gcn_weights_file.exists() or not gcn_scaler_file.exists():
            raise FileNotFoundError(
                f"GCN model files not found in {parent_dir}. "
                f"Expected: bicep_curl_gcn.pth and bicep_curl_gcn_scaler.pkl"
            )

        gcn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gcn_model = _PoseGCN()
        gcn_model.load_state_dict(
            torch.load(gcn_weights_file, map_location=gcn_device)
        )
        gcn_model.to(gcn_device)
        gcn_model.eval()

        with open(gcn_scaler_file, "rb") as f:
            gcn_scaler = pickle.load(f)

        # ── Import arm-analysis logic from bicep_improved (unchanged) ─────
        import bicep_improved as bicep_module
        BicepPoseAnalysis        = bicep_module.BicepPoseAnalysis
        calculate_angle          = bicep_module.calculate_angle
        extract_important_keypoints = bicep_module.extract_important_keypoints

        # POSTURE_ERROR_THRESHOLD: kept at 0.95 to match original behaviour.
        # For the GCN model this means: only update stand_posture when
        # the model's top-class probability is >= 0.95.
        POSTURE_ERROR_THRESHOLD = 0.95

        mp_pose = mp.solutions.pose

        # Landmark names in the order the original bicep_curl.py uses them
        lm_names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
        ]

        # ── Open video ────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS)
        start_time = time.time()

        # ── Arm analysers (same as original) ─────────────────────────────
        left_arm  = BicepPoseAnalysis("LEFT")
        right_arm = BicepPoseAnalysis("RIGHT")

        stand_posture          = "C"
        previous_stand_posture = "C"
        total_reps_left        = 0
        total_reps_right       = 0
        score_left             = 0
        score_right            = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    # ── Build raw landmark dict for GCN ───────────────────
                    landmarks = res.pose_landmarks.landmark
                    raw_row = {}
                    for lm_name in lm_names:
                        lm_key = mp_pose.PoseLandmark[lm_name].value
                        lm_obj = landmarks[lm_key]
                        col    = lm_name.lower()
                        raw_row[f"{col}_x"] = lm_obj.x
                        raw_row[f"{col}_y"] = lm_obj.y
                        raw_row[f"{col}_z"] = lm_obj.z
                        raw_row[f"{col}_v"] = lm_obj.visibility

                    # ── GCN prediction for lean-back (replaces sklearn) ───
                    predicted_posture, ml_confidence = _predict_posture_gcn(
                        raw_row, gcn_model, gcn_scaler, gcn_device
                    )

                    # Same threshold logic as the original file
                    if ml_confidence >= POSTURE_ERROR_THRESHOLD:
                        stand_posture = predicted_posture

                    lean_back_error = (stand_posture == "L")

                    # ── Arm analysis — completely unchanged ───────────────
                    l_done, l_err, _, _ = left_arm.analyze(
                        res.pose_landmarks.landmark, lean_back_error
                    )
                    r_done, r_err, _, _ = right_arm.analyze(
                        res.pose_landmarks.landmark, lean_back_error
                    )

                    if l_done:
                        total_reps_left += 1
                        if lean_back_error:
                            pass  # Bad rep — no score
                        elif l_err:
                            score_left += 5
                        else:
                            score_left += 10

                    if r_done:
                        total_reps_right += 1
                        if lean_back_error:
                            pass  # Bad rep — no score
                        elif r_err:
                            score_right += 5
                        else:
                            score_right += 10

                    previous_stand_posture = stand_posture

        cap.release()
        duration = time.time() - start_time

        # ── Score calculation — unchanged from original ───────────────────
        total_reps      = total_reps_left + total_reps_right
        total_score     = score_left + score_right
        max_score       = total_reps * 10 if total_reps > 0 else 10
        score_percentage = int((total_score / max_score) * 100) if max_score > 0 else 0

        total_errors = (
            left_arm.detected_errors["LOOSE_UPPER_ARM"] +
            left_arm.detected_errors["PEAK_CONTRACTION"] +
            right_arm.detected_errors["LOOSE_UPPER_ARM"] +
            right_arm.detected_errors["PEAK_CONTRACTION"]
        )

        return {
            "score": score_percentage,
            "duration_seconds": duration,
            "error_count": total_errors,
            "details": {
                "left_reps":   left_arm.get_counter(),
                "right_reps":  right_arm.get_counter(),
                "total_reps":  total_reps,
                "left_errors": left_arm.detected_errors,
                "right_errors": right_arm.detected_errors,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# PlankDetector  — completely unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class PlankDetector(ExerciseDetector):
    """Plank exercise detector"""

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process plank video"""
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
        import time
        import pickle

        # Load models from parent directory
        model_file  = parent_dir / "plank_model.pkl"
        scaler_file = parent_dir / "plank_input_scaler.pkl"

        if not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Model files not found in {parent_dir}")

        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)

        # Import detection logic from plank_improved
        import plank_improved as plank_module
        PlankDetection               = plank_module.PlankDetection
        extract_important_keypoints  = plank_module.extract_important_keypoints
        PREDICTION_PROBABILITY_THRESHOLD = plank_module.PREDICTION_PROBABILITY_THRESHOLD

        mp_pose = mp.solutions.pose

        lm_names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        ]

        headers = []
        for lm in lm_names:
            headers += [f"{lm.lower()}_{c}" for c in ["x", "y", "z", "v"]]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps                = cap.get(cv2.CAP_PROP_FPS)
        session_start_time = time.time()

        plank_detector = PlankDetection()

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                current_time = time.time()

                if res.pose_landmarks:
                    row = extract_important_keypoints(res, lm_names)
                    X   = scaler.transform(pd.DataFrame([row], columns=headers))

                    predicted_class    = model.predict(X)[0]
                    prediction_probs   = model.predict_proba(X)[0]
                    max_prob_idx       = np.argmax(prediction_probs)
                    ml_confidence      = float(prediction_probs[max_prob_idx])

                    plank_detector.analyze(predicted_class, ml_confidence, current_time)

        cap.release()
        total_time = time.time() - session_start_time

        error_stats      = plank_detector.get_error_stats()
        total_error_sec  = error_stats["total_error_time"]
        if plank_detector.error_start_time:
            total_error_sec += time.time() - plank_detector.error_start_time

        error_percentage = (total_error_sec / total_time * 100) if total_time > 0 else 0
        quality_score    = max(0, 100 - int(error_percentage))

        total_errors = error_stats["low_back_count"] + error_stats["high_back_count"]

        return {
            "score": quality_score,
            "duration_seconds": total_time,
            "error_count": total_errors,
            "details": {
                "low_back_errors":     error_stats["low_back_count"],
                "high_back_errors":    error_stats["high_back_count"],
                "error_time_seconds":  total_error_sec,
                "error_percentage":    error_percentage,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry — unchanged
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_DETECTORS: Dict[str, ExerciseDetector] = {
    "bicep_curl": BicepCurlDetector(),
    "plank":      PlankDetector(),
}


def get_detector(exercise_name: str) -> ExerciseDetector:
    """Get detector for exercise name"""
    if exercise_name not in EXERCISE_DETECTORS:
        raise ValueError(f"No detector found for exercise: {exercise_name}")
    return EXERCISE_DETECTORS[exercise_name]
