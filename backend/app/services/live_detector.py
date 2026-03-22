"""
live_detector.py
─────────────────────────────────────────────────────────────────────────────
Real-time webcam exercise detection via WebSocket.

Architecture
  Browser  ──JPEG bytes──▶  WebSocket  ──▶  MediaPipe + GCN  ──▶  JSON response
                                                                     {
                                                                       annotated_frame: <base64 JPEG>,
                                                                       stats: { reps, score, errors, … }
                                                                     }

Drop this file alongside exercise_detector.py and add the router to main.py:

    from app.services.live_detector import router as live_router
    app.include_router(live_router)

The WebSocket endpoint will be available at:
    ws://localhost:8000/api/live/{exercise_name}

Supported exercises: bicep_curl, plank
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# ── path setup ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

_backend_dir = Path(__file__).parent.parent.parent  # .../backend/

def _find_project_root() -> Path:
    """
    Walk upward from the backend directory until we find the folder that
    contains the model .pkl files.  This works regardless of whether the
    code is in 'testing02' or 'testing02 - Copy' or any other folder name.
    """
    candidate = _backend_dir.parent  # first guess: sibling of backend/
    for _ in range(4):               # search up to 4 levels up
        if (candidate / "plank_model.pkl").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: return the original guess and let the caller report the error
    return _backend_dir.parent

_parent_dir = _find_project_root()
sys.path.insert(0, str(_parent_dir))

router = APIRouter()

# ── shared GCN imports (only loaded once at module level) ──────────────────
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import pickle

# ═══════════════════════════════════════════════════════════════════════════
#  GCN model definition  (must match training notebook exactly)
# ═══════════════════════════════════════════════════════════════════════════

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
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),   ("right_shoulder", "right_elbow"),
    ("left_elbow",    "left_wrist"),   ("right_elbow",    "right_wrist"),
    ("left_shoulder", "left_hip"),     ("right_shoulder", "right_hip"),
    ("left_hip",      "right_hip"),
]
_node_to_idx = {n: i for i, n in enumerate(_BICEP_LANDMARKS)}
_src, _dst = [], []
for _u, _v in _SKELETON_EDGES:
    _i, _j = _node_to_idx[_u], _node_to_idx[_v]
    _src += [_i, _j]; _dst += [_j, _i]
_EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)

_BICEP_FEATURE_COLS = [
    f"{lm}_{c}" for lm in _BICEP_LANDMARKS for c in ["x", "y", "z", "v"]
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
    x_node      = torch.tensor(feat_scaled[0].reshape(_N_NODES, _N_FEATS), dtype=torch.float)
    data        = Data(x=x_node, edge_index=_EDGE_INDEX)
    data.batch  = torch.zeros(_N_NODES, dtype=torch.long)
    data        = data.to(device)
    with torch.no_grad():
        logits = gcn_model(data)
        probs  = F.softmax(logits, dim=1)[0]
    prob_c, prob_l = probs[0].item(), probs[1].item()
    confidence     = max(prob_c, prob_l)
    predicted      = "L" if prob_l >= (1 - threshold) else "C"
    return predicted, confidence


# ═══════════════════════════════════════════════════════════════════════════
#  Inlined BicepPoseAnalysis  (no dependency on bicep_improved.py)
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_angle(p1, p2, p3) -> float:
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return deg if deg <= 180 else 360 - deg


class _BicepPoseAnalysis:
    STAGE_DOWN_THRESHOLD       = 120
    STAGE_UP_THRESHOLD         = 100
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_THRESHOLD  = 40
    VISIBILITY_THRESHOLD       = 0.65

    def __init__(self, side: str):
        self._mp_pose = mp.solutions.pose
        self.side     = side
        self.counter  = 0
        self.stage    = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm        = False
        self.peak_contraction_angle = 1000
        self.shoulder = self.elbow = self.wrist = None

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

    def analyze_pose(self, landmarks, lean_back_error=False):
        if not self._get_joints(landmarks):
            return None, None, False
        curl_angle = int(_calculate_angle(self.shoulder, self.elbow, self.wrist))
        if curl_angle > self.STAGE_DOWN_THRESHOLD:
            self.stage = "down"
        elif curl_angle < self.STAGE_UP_THRESHOLD and self.stage == "down":
            self.stage   = "up"
            self.counter += 1
        shoulder_projection = [self.shoulder[0], 1]
        ground_angle = int(_calculate_angle(self.elbow, self.shoulder, shoulder_projection))
        has_error = False
        if lean_back_error:
            return curl_angle, ground_angle, False
        if ground_angle > self.LOOSE_UPPER_ARM_THRESHOLD:
            has_error = True
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
        if self.stage == "up" and curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = curl_angle
        elif self.stage == "down":
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.PEAK_CONTRACTION_THRESHOLD:
                self.detected_errors["PEAK_CONTRACTION"] += 1
                has_error = True
            self.peak_contraction_angle = 1000
        return curl_angle, ground_angle, has_error

    def get_counter(self):
        return self.counter


# ═══════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

# Colour palette (BGR)
_C_GREEN  = (0,   220, 100)
_C_RED    = (0,    60, 220)
_C_YELLOW = (0,   200, 255)
_C_WHITE  = (255, 255, 255)
_C_BLACK  = (0,     0,   0)
_C_ACCENT = (255, 180,   0)   # electric blue-ish in BGR

_mp_drawing         = mp.solutions.drawing_utils
_mp_drawing_styles  = mp.solutions.drawing_styles
_mp_pose_module     = mp.solutions.pose

_LANDMARK_SPEC = _mp_drawing.DrawingSpec(color=_C_GREEN, thickness=2, circle_radius=4)
_CONNECTION_SPEC = _mp_drawing.DrawingSpec(color=(180, 255, 180), thickness=2)


def _draw_skeleton(frame, landmarks):
    """Draw MediaPipe pose skeleton on the frame."""
    _mp_drawing.draw_landmarks(
        frame,
        landmarks,
        _mp_pose_module.POSE_CONNECTIONS,
        landmark_drawing_spec=_LANDMARK_SPEC,
        connection_drawing_spec=_CONNECTION_SPEC,
    )


def _draw_hud(frame, stats: dict, exercise: str):
    """Draw a clean HUD overlay with stats."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Exercise label
    cv2.putText(frame, exercise.upper().replace("_", " "),
                (16, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, _C_ACCENT, 1, cv2.LINE_AA)

    if exercise == "bicep_curl":
        reps  = stats.get("total_reps", 0)
        score = stats.get("score", 0)
        errs  = stats.get("total_errors", 0)
        lean  = stats.get("lean_back", False)

        # Reps pill
        _pill(frame, f"REPS  {reps}", (16, 38), _C_GREEN)
        # Score pill
        score_col = _C_GREEN if score >= 80 else _C_YELLOW if score >= 50 else _C_RED
        _pill(frame, f"SCORE  {score}", (110, 38), score_col)
        # Errors pill
        _pill(frame, f"ERRORS  {errs}", (220, 38), _C_RED if errs > 0 else _C_WHITE)

        # Active error banners
        active_errors = stats.get("active_errors", [])
        for i, err in enumerate(active_errors):
            _error_banner(frame, err, i, w, h)

        if lean:
            _error_banner(frame, "LEAN BACK DETECTED", 0, w, h)

    elif exercise == "plank":
        score   = stats.get("score", 100)
        cur_err = stats.get("current_error", None)
        dur     = stats.get("duration", 0)

        _pill(frame, f"HOLD  {int(dur)}s", (16, 38), _C_GREEN)
        score_col = _C_GREEN if score >= 80 else _C_YELLOW if score >= 50 else _C_RED
        _pill(frame, f"QUALITY  {score}%", (110, 38), score_col)

        if cur_err:
            _error_banner(frame, cur_err, 0, w, h)


def _pill(frame, text, pos, color):
    """Draw a small filled pill label."""
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), color, -1)
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), _C_BLACK, 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_BLACK, 1, cv2.LINE_AA)


def _error_banner(frame, text, idx, w, h):
    """Draw a red error banner near the bottom."""
    y = h - 40 - idx * 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 28), (w, y + 8), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"⚠  {text}", (16, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, _C_WHITE, 1, cv2.LINE_AA)


def _encode_frame(frame, quality=80) -> str:
    """Encode OpenCV frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  Session state containers
# ═══════════════════════════════════════════════════════════════════════════

class _BicepSession:
    """Holds all mutable state for a single live bicep curl session."""

    def __init__(self, gcn_model, scaler, device):
        self.gcn_model    = gcn_model
        self.scaler       = scaler
        self.device       = device
        self.left_arm     = _BicepPoseAnalysis("LEFT")
        self.right_arm    = _BicepPoseAnalysis("RIGHT")
        self.stand_posture = "C"
        self.score_left   = 0
        self.score_right  = 0
        self.prev_left    = 0
        self.prev_right   = 0
        self.start_time   = time.time()

    def process_landmarks(self, landmarks, lm_names, mp_pose):
        # Build raw row
        raw_row = {}
        for lm_name in lm_names:
            lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
            col    = lm_name.lower()
            raw_row[f"{col}_x"] = lm_obj.x
            raw_row[f"{col}_y"] = lm_obj.y
            raw_row[f"{col}_z"] = lm_obj.z
            raw_row[f"{col}_v"] = lm_obj.visibility

        # GCN posture
        predicted, confidence = _predict_posture_gcn(
            raw_row, self.gcn_model, self.scaler, self.device
        )
        if confidence >= 0.95:
            self.stand_posture = predicted

        lean_back = (self.stand_posture == "L")

        # Arm analysis
        _, _, left_err  = self.left_arm.analyze_pose(landmarks, lean_back)
        _, _, right_err = self.right_arm.analyze_pose(landmarks, lean_back)

        # Score on new rep
        new_left  = self.left_arm.get_counter()
        new_right = self.right_arm.get_counter()
        if new_left > self.prev_left:
            if not lean_back:
                self.score_left += 5 if left_err else 10
            self.prev_left = new_left
        if new_right > self.prev_right:
            if not lean_back:
                self.score_right += 5 if right_err else 10
            self.prev_right = new_right

        total_reps  = new_left + new_right
        total_score = self.score_left + self.score_right
        max_score   = total_reps * 10 if total_reps > 0 else 10
        score_pct   = int((total_score / max_score) * 100) if max_score > 0 else 0

        active_errors = []
        if left_err or right_err:
            if (self.left_arm.loose_upper_arm or self.right_arm.loose_upper_arm):
                active_errors.append("LOOSE UPPER ARM")
            if (self.left_arm.peak_contraction_angle != 1000 or
                    self.right_arm.peak_contraction_angle != 1000):
                active_errors.append("INCOMPLETE PEAK CONTRACTION")

        total_errors = (
            self.left_arm.detected_errors["LOOSE_UPPER_ARM"] +
            self.left_arm.detected_errors["PEAK_CONTRACTION"] +
            self.right_arm.detected_errors["LOOSE_UPPER_ARM"] +
            self.right_arm.detected_errors["PEAK_CONTRACTION"]
        )

        return {
            "total_reps"   : total_reps,
            "left_reps"    : new_left,
            "right_reps"   : new_right,
            "score"        : score_pct,
            "total_errors" : total_errors,
            "active_errors": active_errors,
            "lean_back"    : lean_back,
            "duration"     : time.time() - self.start_time,
        }


class _PlankSession:
    """Holds mutable state for a live plank session (uses sklearn model)."""

    def __init__(self, model, scaler):
        self.model       = model
        self.scaler      = scaler
        self.start_time  = time.time()
        self.error_time  = 0.0
        self.error_start = None
        self.low_back    = 0
        self.high_back   = 0
        self.current_err = None

    def process_row(self, row, headers):
        import pandas as pd
        X     = self.scaler.transform(pd.DataFrame([row], columns=headers))
        pred  = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        conf  = float(probs[np.argmax(probs)])

        THRESHOLD = 0.6
        now = time.time()

        if conf < THRESHOLD:
            # below threshold — treat as correct
            if self.error_start is not None:
                self.error_time += now - self.error_start
                self.error_start = None
            self.current_err = None
        elif pred == "L":
            if self.error_start is None:
                self.error_start = now
                self.low_back += 1
            self.current_err = "LOW BACK — RAISE YOUR HIPS"
        elif pred == "H":
            if self.error_start is None:
                self.error_start = now
                self.high_back += 1
            self.current_err = "HIGH BACK — LOWER YOUR HIPS"
        else:
            if self.error_start is not None:
                self.error_time += now - self.error_start
                self.error_start = None
            self.current_err = None

        duration = now - self.start_time
        total_err_sec = self.error_time
        if self.error_start:
            total_err_sec += now - self.error_start
        err_pct = (total_err_sec / duration * 100) if duration > 0 else 0
        quality = max(0, 100 - int(err_pct))

        return {
            "duration"     : duration,
            "score"        : quality,
            "low_back"     : self.low_back,
            "high_back"    : self.high_back,
            "current_error": self.current_err,
            "error_pct"    : round(err_pct, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.websocket("/api/live/bicep_curl")
async def live_bicep_curl(websocket: WebSocket):
    """
    WebSocket for live bicep curl detection.
    
    Protocol:
      Client → Server : raw JPEG bytes (one frame at a time)
      Server → Client : JSON string  {
                            annotated_frame: <base64 JPEG>,
                            stats: { … }
                          }
    """
    await websocket.accept()

    # Load GCN model
    gcn_weights = _parent_dir / "bicep_curl_gcn.pth"
    gcn_scaler  = _parent_dir / "bicep_curl_gcn_scaler.pkl"
    if not gcn_weights.exists() or not gcn_scaler.exists():
        await websocket.send_text(json.dumps({
            "error": f"GCN model files not found. Expected bicep_curl_gcn.pth and bicep_curl_gcn_scaler.pkl in {_parent_dir}"
        }))
        await websocket.close()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn_model = _PoseGCN()
    gcn_model.load_state_dict(torch.load(gcn_weights, map_location=device))
    gcn_model.to(device).eval()
    with open(gcn_scaler, "rb") as f:
        scaler = pickle.load(f)

    mp_pose  = mp.solutions.pose
    lm_names = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "RIGHT_ELBOW", "LEFT_ELBOW",
        "RIGHT_WRIST", "LEFT_WRIST",
        "LEFT_HIP", "RIGHT_HIP",
    ]

    session = _BicepSession(gcn_model, scaler, device)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as pose:
            while True:
                # Receive JPEG bytes from browser
                data = await websocket.receive_bytes()

                # Decode
                nparr  = np.frombuffer(data, np.uint8)
                frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Process with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                stats = {
                    "total_reps": 0, "left_reps": 0, "right_reps": 0,
                    "score": 0, "total_errors": 0, "active_errors": [],
                    "lean_back": False, "duration": time.time() - session.start_time,
                }

                if res.pose_landmarks:
                    _draw_skeleton(frame, res.pose_landmarks)
                    stats = session.process_landmarks(res.pose_landmarks.landmark, lm_names, mp_pose)

                _draw_hud(frame, stats, "bicep_curl")

                b64 = _encode_frame(frame)
                await websocket.send_text(json.dumps({
                    "annotated_frame": b64,
                    "stats": stats,
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


@router.websocket("/api/live/plank")
async def live_plank(websocket: WebSocket):
    """
    WebSocket for live plank detection.
    Uses the existing sklearn plank model (plank_model.pkl / plank_input_scaler.pkl).
    """
    await websocket.accept()

    model_file  = _parent_dir / "plank_model.pkl"
    scaler_file = _parent_dir / "plank_input_scaler.pkl"
    if not model_file.exists() or not scaler_file.exists():
        await websocket.send_text(json.dumps({
            "error": f"Plank model files not found in {_parent_dir}"
        }))
        await websocket.close()
        return

    with open(model_file,  "rb") as f: model  = pickle.load(f)
    with open(scaler_file, "rb") as f: scaler = pickle.load(f)

    # Load plank_improved using importlib with explicit file path — this is
    # robust regardless of sys.path state and never touches exercise_detector.py.
    plank_script = _parent_dir / "plank_improved.py"
    if not plank_script.exists():
        await websocket.send_text(json.dumps({
            "error": f"plank_improved.py not found at {plank_script}. "
                     f"Make sure it exists in the project root alongside the .pkl files."
        }))
        await websocket.close()
        return

    import importlib.util
    _spec   = importlib.util.spec_from_file_location("plank_improved", plank_script)
    _mod    = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    extract_kp = _mod.extract_important_keypoints

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
    headers = [f"{lm.lower()}_{c}" for lm in lm_names for c in ["x", "y", "z", "v"]]

    session = _PlankSession(model, scaler)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as pose:
            while True:
                data  = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                stats = {
                    "duration": time.time() - session.start_time,
                    "score": 100, "low_back": 0, "high_back": 0,
                    "current_error": None, "error_pct": 0.0,
                }

                if res.pose_landmarks:
                    _draw_skeleton(frame, res.pose_landmarks)
                    row   = extract_kp(res, lm_names)
                    stats = session.process_row(row, headers)

                _draw_hud(frame, stats, "plank")

                b64 = _encode_frame(frame)
                await websocket.send_text(json.dumps({
                    "annotated_frame": b64,
                    "stats": stats,
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass