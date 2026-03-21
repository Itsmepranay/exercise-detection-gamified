from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

backend_dir = Path(__file__).parent.parent.parent
parent_dir  = backend_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared geometry helpers — inlined, no external dependency
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_angle(point1: list, point2: list, point3: list) -> float:
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    rad = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return deg if deg <= 180 else 360 - deg


# ─────────────────────────────────────────────────────────────────────────────
# _BicepPoseAnalysis
# Inlined from web/server/detection/bicep_curl.py — BicepPoseAnalysis class.
# Tracks rep count + detects LOOSE_UPPER_ARM and PEAK_CONTRACTION per arm.
# The only change: removed cv2 drawing calls and results.append() side effects
# since this is used in the backend video processor, not a live display loop.
# ─────────────────────────────────────────────────────────────────────────────

class _BicepPoseAnalysis:
    STAGE_DOWN_THRESHOLD       = 120
    STAGE_UP_THRESHOLD         = 100
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_THRESHOLD  = 40
    VISIBILITY_THRESHOLD       = 0.65

    def __init__(self, side: str):
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self.side     = side

        self.counter    = 0
        self.stage      = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM" : 0,
            "PEAK_CONTRACTION": 0,
        }
        self.loose_upper_arm        = False
        self.peak_contraction_angle = 1000
        self.shoulder = None
        self.elbow    = None
        self.wrist    = None

    def _get_joints(self, landmarks) -> bool:
        side = self.side.upper()
        vis  = [
            landmarks[self._mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[self._mp_pose.PoseLandmark[f"{side}_ELBOW"   ].value].visibility,
            landmarks[self._mp_pose.PoseLandmark[f"{side}_WRIST"   ].value].visibility,
        ]
        self.is_visible = all(v > self.VISIBILITY_THRESHOLD for v in vis)
        if not self.is_visible:
            return False

        def xy(name):
            lm = landmarks[self._mp_pose.PoseLandmark[name].value]
            return [lm.x, lm.y]

        self.shoulder = xy(f"{side}_SHOULDER")
        self.elbow    = xy(f"{side}_ELBOW")
        self.wrist    = xy(f"{side}_WRIST")
        return True

    def analyze_pose(self, landmarks, lean_back_error: bool = False) -> tuple:
        """
        Analyse one frame for this arm.
        Returns (curl_angle, ground_upper_arm_angle, has_error).
        Returns (None, None, False) if arm is not visible.
        """
        has_error = False

        if not self._get_joints(landmarks):
            return (None, None, False)

        # Rep counter
        curl_angle = int(_calculate_angle(self.shoulder, self.elbow, self.wrist))
        if curl_angle > self.STAGE_DOWN_THRESHOLD:
            self.stage = "down"
        elif curl_angle < self.STAGE_UP_THRESHOLD and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        # Upper-arm angle vs vertical
        shoulder_projection = [self.shoulder[0], 1]
        ground_angle = int(
            _calculate_angle(self.elbow, self.shoulder, shoulder_projection)
        )

        # Skip arm-error checks when lean-back is active
        if lean_back_error:
            return (curl_angle, ground_angle, False)

        # LOOSE UPPER ARM
        if ground_angle > self.LOOSE_UPPER_ARM_THRESHOLD:
            has_error = True
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False

        # PEAK CONTRACTION
        if self.stage == "up" and curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = curl_angle
        elif self.stage == "down":
            if (
                self.peak_contraction_angle != 1000
                and self.peak_contraction_angle >= self.PEAK_CONTRACTION_THRESHOLD
            ):
                self.detected_errors["PEAK_CONTRACTION"] += 1
                has_error = True
            self.peak_contraction_angle = 1000

        return (curl_angle, ground_angle, has_error)

    def get_counter(self) -> int:
        return self.counter

    def reset(self):
        self.counter    = 0
        self.stage      = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm        = False
        self.peak_contraction_angle = 1000


# ─────────────────────────────────────────────────────────────────────────────
# GCN model — must match training notebook exactly
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

_BICEP_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "right_elbow",    "left_elbow",
    "right_wrist",    "left_wrist",
    "left_hip",       "right_hip",
]
_N_NODES = len(_BICEP_LANDMARKS)
_N_FEATS = 4

_SKELETON_EDGES = [
    ("nose","left_shoulder"), ("nose","right_shoulder"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),  ("right_shoulder","right_elbow"),
    ("left_elbow","left_wrist"),     ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),    ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
]
_node_to_idx = {n: i for i, n in enumerate(_BICEP_LANDMARKS)}
_src, _dst = [], []
for _u, _v in _SKELETON_EDGES:
    _i, _j = _node_to_idx[_u], _node_to_idx[_v]
    _src += [_i, _j]; _dst += [_j, _i]
_EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)

_BICEP_FEATURE_COLS = [
    f"{lm}_{c}" for lm in _BICEP_LANDMARKS for c in ["x","y","z","v"]
]


class _PoseGCN(nn.Module):
    def __init__(self, in_feats=4, hidden=64, out_feats=32, n_classes=2, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(out_feats)
        self.head = nn.Sequential(
            nn.Linear(_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = x.view(batch.max().item() + 1, _N_NODES * self.out_feats)
        return self.head(x)


def _normalize_pose_row(row: dict) -> dict:
    cx = (row["left_shoulder_x"] + row["right_shoulder_x"] +
          row["left_hip_x"]      + row["right_hip_x"]) / 4
    cy = (row["left_shoulder_y"] + row["right_shoulder_y"] +
          row["left_hip_y"]      + row["right_hip_y"]) / 4
    sh_mx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sh_my = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hi_mx = (row["left_hip_x"]      + row["right_hip_x"])      / 2
    hi_my = (row["left_hip_y"]      + row["right_hip_y"])      / 2
    torso = ((sh_mx - hi_mx)**2 + (sh_my - hi_my)**2)**0.5 + 1e-6
    out = dict(row)
    for lm in _BICEP_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - cx) / torso
        out[f"{lm}_y"] = (row[f"{lm}_y"] - cy) / torso
        out[f"{lm}_z"] =  row[f"{lm}_z"]        / torso
    return out


def _predict_posture_gcn(raw_row, gcn_model, scaler, device, threshold=0.95):
    norm        = _normalize_pose_row(raw_row)
    feat_vec    = [[norm[col] for col in _BICEP_FEATURE_COLS]]
    feat_scaled = scaler.transform(feat_vec)
    x_node = torch.tensor(feat_scaled[0].reshape(_N_NODES, _N_FEATS), dtype=torch.float)
    data        = Data(x=x_node, edge_index=_EDGE_INDEX)
    data.batch  = torch.zeros(_N_NODES, dtype=torch.long)
    data        = data.to(device)
    with torch.no_grad():
        logits = gcn_model(data)
        probs  = F.softmax(logits, dim=1)[0]
    prob_c, prob_l = probs[0].item(), probs[1].item()
    confidence      = max(prob_c, prob_l)
    predicted_class = "L" if prob_l >= (1 - threshold) else "C"
    return predicted_class, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ExerciseDetector(ABC):
    @abstractmethod
    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# BicepCurlDetector
# ─────────────────────────────────────────────────────────────────────────────

class BicepCurlDetector(ExerciseDetector):
    """
    Bicep curl detector.
      Lean-back detection : GCN (bicep_curl_gcn.pth + bicep_curl_gcn_scaler.pkl)
      Arm analysis        : _BicepPoseAnalysis (inlined above — no bicep_improved.py)
    """

    POSTURE_ERROR_THRESHOLD = 0.95

    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import time
        import pickle

        # Load GCN
        gcn_weights = parent_dir / "bicep_curl_gcn.pth"
        gcn_scaler  = parent_dir / "bicep_curl_gcn_scaler.pkl"
        if not gcn_weights.exists() or not gcn_scaler.exists():
            raise FileNotFoundError(
                f"GCN files not found in {parent_dir}. "
                "Expected: bicep_curl_gcn.pth and bicep_curl_gcn_scaler.pkl"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gcn_model = _PoseGCN()
        gcn_model.load_state_dict(torch.load(gcn_weights, map_location=device))
        gcn_model.to(device).eval()
        with open(gcn_scaler, "rb") as f:
            scaler = pickle.load(f)

        mp_pose = mp.solutions.pose

        # Landmark names for building the GCN input dict
        lm_names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
        ]

        # Arm analysers
        left_arm  = _BicepPoseAnalysis("LEFT")
        right_arm = _BicepPoseAnalysis("RIGHT")

        stand_posture    = "C"
        score_left       = 0
        score_right      = 0
        prev_left_count  = 0
        prev_right_count = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        start_time = time.time()

        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.8, min_tracking_confidence=0.8,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if not res.pose_landmarks:
                    continue

                landmarks = res.pose_landmarks.landmark

                # Build raw landmark dict for GCNs
                raw_row = {}
                for lm_name in lm_names:
                    lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
                    col    = lm_name.lower()
                    raw_row[f"{col}_x"] = lm_obj.x
                    raw_row[f"{col}_y"] = lm_obj.y
                    raw_row[f"{col}_z"] = lm_obj.z
                    raw_row[f"{col}_v"] = lm_obj.visibility

                # GCN lean-back prediction
                predicted_posture, ml_confidence = _predict_posture_gcn(
                    raw_row, gcn_model, scaler, device
                )
                if ml_confidence >= self.POSTURE_ERROR_THRESHOLD:
                    stand_posture = predicted_posture

                lean_back_error = (stand_posture == "L")

                # Arm analysis
                _, _, left_arm_error  = left_arm.analyze_pose(landmarks, lean_back_error)
                _, _, right_arm_error = right_arm.analyze_pose(landmarks, lean_back_error)

                # Score when a new rep completes (counter delta)
                new_left  = left_arm.get_counter()
                new_right = right_arm.get_counter()

                if new_left > prev_left_count:
                    if not lean_back_error:
                        score_left += 5 if left_arm_error else 10
                    prev_left_count = new_left

                if new_right > prev_right_count:
                    if not lean_back_error:
                        score_right += 5 if right_arm_error else 10
                    prev_right_count = new_right

        cap.release()
        duration = time.time() - start_time

        total_reps   = left_arm.get_counter() + right_arm.get_counter()
        total_score  = score_left + score_right
        max_score    = total_reps * 10 if total_reps > 0 else 10
        score_pct    = int((total_score / max_score) * 100) if max_score > 0 else 0
        total_errors = (
            left_arm.detected_errors["LOOSE_UPPER_ARM"]  +
            left_arm.detected_errors["PEAK_CONTRACTION"] +
            right_arm.detected_errors["LOOSE_UPPER_ARM"] +
            right_arm.detected_errors["PEAK_CONTRACTION"]
        )

        return {
            "score"           : score_pct,
            "duration_seconds": duration,
            "error_count"     : total_errors,
            "details": {
                "left_reps"   : left_arm.get_counter(),
                "right_reps"  : right_arm.get_counter(),
                "total_reps"  : total_reps,
                "left_errors" : left_arm.detected_errors,
                "right_errors": right_arm.detected_errors,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# PlankDetector — completely unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class PlankDetector(ExerciseDetector):
    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import pandas as pd
        import time
        import pickle

        model_file  = parent_dir / "plank_model.pkl"
        scaler_file = parent_dir / "plank_input_scaler.pkl"
        if not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Plank model files not found in {parent_dir}")

        with open(model_file,  "rb") as f: model  = pickle.load(f)
        with open(scaler_file, "rb") as f: scaler = pickle.load(f)

        import plank_improved as plank_module
        PlankDetection               = plank_module.PlankDetection
        extract_important_keypoints  = plank_module.extract_important_keypoints
        PREDICTION_PROBABILITY_THRESHOLD = plank_module.PREDICTION_PROBABILITY_THRESHOLD

        mp_pose  = mp.solutions.pose
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
        headers = [f"{lm.lower()}_{c}" for lm in lm_names for c in ["x","y","z","v"]]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        session_start_time = time.time()
        plank_detector     = PlankDetection()

        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.8, min_tracking_confidence=0.8,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                current_time = time.time()
                if res.pose_landmarks:
                    row   = extract_important_keypoints(res, lm_names)
                    X     = scaler.transform(pd.DataFrame([row], columns=headers))
                    pred  = model.predict(X)[0]
                    probs = model.predict_proba(X)[0]
                    conf  = float(probs[np.argmax(probs)])
                    plank_detector.analyze(pred, conf, current_time)

        cap.release()
        total_time = time.time() - session_start_time

        error_stats     = plank_detector.get_error_stats()
        total_error_sec = error_stats["total_error_time"]
        if plank_detector.error_start_time:
            total_error_sec += time.time() - plank_detector.error_start_time

        error_pct     = (total_error_sec / total_time * 100) if total_time > 0 else 0
        quality_score = max(0, 100 - int(error_pct))
        total_errors  = error_stats["low_back_count"] + error_stats["high_back_count"]

        return {
            "score"           : quality_score,
            "duration_seconds": total_time,
            "error_count"     : total_errors,
            "details": {
                "low_back_errors"   : error_stats["low_back_count"],
                "high_back_errors"  : error_stats["high_back_count"],
                "error_time_seconds": total_error_sec,
                "error_percentage"  : error_pct,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_DETECTORS: Dict[str, ExerciseDetector] = {
    "bicep_curl": BicepCurlDetector(),
    "plank"     : PlankDetector(),
}


def get_detector(exercise_name: str) -> ExerciseDetector:
    if exercise_name not in EXERCISE_DETECTORS:
        raise ValueError(f"No detector found for exercise: {exercise_name}")
    return EXERCISE_DETECTORS[exercise_name]
