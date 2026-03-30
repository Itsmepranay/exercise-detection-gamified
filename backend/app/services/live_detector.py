"""
live_detector.py
─────────────────────────────────────────────────────────────────────────────
Real-time webcam exercise detection via WebSocket.
100% self-contained — no plank_improved.py, no bicep_improved.py.
All detection logic is inlined here.

Add to main.py:
    from app.services.live_detector import router as live_router
    app.include_router(live_router)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import json
import time
import pickle
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ── Project root: walk up until we find the .pkl files ────────────────────
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(8):
        if (candidate / "plank_model.pkl").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        f"Cannot find project root (looked for plank_model.pkl). "
        f"Started from {Path(__file__).resolve()}"
    )

_ROOT = _find_project_root()

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

def _angle(p1, p2, p3) -> float:
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return deg if deg <= 180 else 360 - deg


# ═══════════════════════════════════════════════════════════════════════════
#  BICEP — GCN model (identical to exercise_detector._PoseGCN)
# ═══════════════════════════════════════════════════════════════════════════

_BICEP_LMS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "right_elbow",    "left_elbow",
    "right_wrist",    "left_wrist",
    "left_hip",       "right_hip",
]
_N_NODES = len(_BICEP_LMS)
_N_FEATS = 4

_EDGES = [
    ("nose","left_shoulder"), ("nose","right_shoulder"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),   ("right_shoulder","right_elbow"),
    ("left_elbow","left_wrist"),      ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),     ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
]
_idx = {n: i for i, n in enumerate(_BICEP_LMS)}
_src, _dst = [], []
for _u, _v in _EDGES:
    _i, _j = _idx[_u], _idx[_v]
    _src += [_i, _j]; _dst += [_j, _i]
_EDGE_INDEX = torch.tensor([_src, _dst], dtype=torch.long)
_FEAT_COLS  = [f"{lm}_{c}" for lm in _BICEP_LMS for c in ["x","y","z","v"]]


class _PoseGCN(nn.Module):
    def __init__(self, in_feats=4, hidden=64, out_feats=32, n_classes=2, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(out_feats)
        self.head  = nn.Sequential(
            nn.Linear(_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, ei)));  x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.bn2(self.conv2(x, ei)));  x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.bn3(self.conv3(x, ei)))
        x = x.view(batch.max().item() + 1, _N_NODES * self.out_feats)
        return self.head(x)


def _normalize_row(row: dict) -> dict:
    cx = (row["left_shoulder_x"] + row["right_shoulder_x"] + row["left_hip_x"]  + row["right_hip_x"])  / 4
    cy = (row["left_shoulder_y"] + row["right_shoulder_y"] + row["left_hip_y"]  + row["right_hip_y"])  / 4
    sh_x = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sh_y = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hi_x = (row["left_hip_x"]      + row["right_hip_x"])      / 2
    hi_y = (row["left_hip_y"]      + row["right_hip_y"])      / 2
    torso = ((sh_x - hi_x)**2 + (sh_y - hi_y)**2)**0.5 + 1e-6
    out = dict(row)
    for lm in _BICEP_LMS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - cx) / torso
        out[f"{lm}_y"] = (row[f"{lm}_y"] - cy) / torso
        out[f"{lm}_z"] =  row[f"{lm}_z"]        / torso
    return out


def _gcn_predict(raw_row, model, scaler, device, threshold=0.95):
    norm        = _normalize_row(raw_row)
    feat_scaled = scaler.transform([[norm[c] for c in _FEAT_COLS]])
    x_node      = torch.tensor(feat_scaled[0].reshape(_N_NODES, _N_FEATS), dtype=torch.float)
    data        = Data(x=x_node, edge_index=_EDGE_INDEX)
    data.batch  = torch.zeros(_N_NODES, dtype=torch.long)
    data        = data.to(device)
    with torch.no_grad():
        probs = F.softmax(model(data), dim=1)[0]
    prob_c, prob_l = probs[0].item(), probs[1].item()
    return ("L" if prob_l >= (1 - threshold) else "C"), max(prob_c, prob_l)


# ═══════════════════════════════════════════════════════════════════════════
#  BICEP — arm analyser (identical to exercise_detector._BicepPoseAnalysis)
# ═══════════════════════════════════════════════════════════════════════════

class _ArmAnalyser:
    DOWN_TH   = 120;  UP_TH   = 100
    PEAK_TH   = 60;   LOOSE_TH = 40
    VIS_TH    = 0.65

    def __init__(self, side: str):
        self._pose   = mp.solutions.pose
        self.side    = side.upper()
        self.counter = 0
        self.stage   = "down"
        self.detected_errors     = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm     = False
        self.peak_angle          = 1000
        self.shoulder = self.elbow = self.wrist = None

    def _joints(self, lms) -> bool:
        s = self.side
        vis = [lms[self._pose.PoseLandmark[f"{s}_SHOULDER"].value].visibility,
               lms[self._pose.PoseLandmark[f"{s}_ELBOW"   ].value].visibility,
               lms[self._pose.PoseLandmark[f"{s}_WRIST"   ].value].visibility]
        if not all(v > self.VIS_TH for v in vis):
            return False
        def xy(n): lm = lms[self._pose.PoseLandmark[n].value]; return [lm.x, lm.y]
        self.shoulder = xy(f"{s}_SHOULDER")
        self.elbow    = xy(f"{s}_ELBOW")
        self.wrist    = xy(f"{s}_WRIST")
        return True

    def analyse(self, lms, lean_back=False):
        if not self._joints(lms):
            return None, None, False
        curl = int(_angle(self.shoulder, self.elbow, self.wrist))
        if curl > self.DOWN_TH:
            self.stage = "down"
        elif curl < self.UP_TH and self.stage == "down":
            self.stage = "up"; self.counter += 1
        proj  = [self.shoulder[0], 1]
        ground = int(_angle(self.elbow, self.shoulder, proj))
        if lean_back:
            return curl, ground, False
        err = False
        if ground > self.LOOSE_TH:
            err = True
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
        if self.stage == "up" and curl < self.peak_angle:
            self.peak_angle = curl
        elif self.stage == "down":
            if self.peak_angle != 1000 and self.peak_angle >= self.PEAK_TH:
                self.detected_errors["PEAK_CONTRACTION"] += 1; err = True
            self.peak_angle = 1000
        return curl, ground, err


# ═══════════════════════════════════════════════════════════════════════════
#  PLANK — keypoint extractor (replaces plank_improved.extract_important_keypoints)
# ═══════════════════════════════════════════════════════════════════════════

_PLANK_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
_PLANK_HEADERS = [f"{lm.lower()}_{c}" for lm in _PLANK_LMS for c in ["x","y","z","v"]]


def _extract_plank_keypoints(results) -> list:
    """Flatten pose landmarks into a flat list matching _PLANK_HEADERS."""
    lms = results.pose_landmarks.landmark
    mp_pose = mp.solutions.pose
    row = []
    for lm_name in _PLANK_LMS:
        lm = lms[mp_pose.PoseLandmark[lm_name].value]
        row += [lm.x, lm.y, lm.z, lm.visibility]
    return row


# ═══════════════════════════════════════════════════════════════════════════
#  PLANK — error tracker (replaces plank_improved.PlankDetection)
# ═══════════════════════════════════════════════════════════════════════════

class _PlankTracker:
    """
    Tracks plank error timing.
    Mirrors the PlankDetection.analyze() / get_error_stats() interface
    used in exercise_detector.PlankDetector.process_video().
    Labels: "C" = correct, "L" = low back, "H" = high back.
    """
    THRESHOLD = 0.60

    def __init__(self):
        self.error_start_time = None
        self._total_error     = 0.0
        self._low_back        = 0
        self._high_back       = 0
        self._last_pred       = "C"

    def analyze(self, pred: str, conf: float, now: float):
        self._last_pred = pred
        is_error = conf >= self.THRESHOLD and pred in ("L", "H")

        if is_error:
            if self.error_start_time is None:
                self.error_start_time = now
                if pred == "L": self._low_back  += 1
                else:           self._high_back += 1
        else:
            if self.error_start_time is not None:
                self._total_error    += now - self.error_start_time
                self.error_start_time = None

    def get_error_stats(self) -> dict:
        return {
            "total_error_time": self._total_error,
            "low_back_count"  : self._low_back,
            "high_back_count" : self._high_back,
        }

    def current_error_label(self, pred: str, conf: float) -> str | None:
        if self.error_start_time is None:
            return None
        if pred == "L" and conf >= self.THRESHOLD:
            return "LOW BACK — RAISE YOUR HIPS"
        if pred == "H" and conf >= self.THRESHOLD:
            return "HIGH BACK — LOWER YOUR HIPS"
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════

_mp_draw   = mp.solutions.drawing_utils
_mp_pose_m = mp.solutions.pose
_LM_SPEC   = _mp_draw.DrawingSpec(color=(0, 220, 100), thickness=2, circle_radius=4)
_CN_SPEC   = _mp_draw.DrawingSpec(color=(180, 255, 180), thickness=2)


def _draw_skeleton(frame, landmarks):
    _mp_draw.draw_landmarks(frame, landmarks, _mp_pose_m.POSE_CONNECTIONS,
                            _LM_SPEC, _CN_SPEC)


def _pill(frame, text, pos, color):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), color, -1)
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), (0,0,0), 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1, cv2.LINE_AA)


def _banner(frame, text, idx, w, h):
    y = h - 40 - idx * 36
    ov = frame.copy()
    cv2.rectangle(ov, (0, y-28), (w, y+8), (0, 0, 180), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"!  {text}", (16, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)


def _hud(frame, stats: dict, exercise: str):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,70), (15,15,15), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, exercise.upper().replace("_"," "),
                (16,24), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255,180,0), 1, cv2.LINE_AA)

    if exercise == "bicep_curl":
        reps  = stats.get("total_reps", 0)
        score = stats.get("score", 0)
        errs  = stats.get("total_errors", 0)
        _pill(frame, f"REPS  {reps}",   (16, 38), (0,220,100))
        sc = (0,220,100) if score>=80 else (0,200,255) if score>=50 else (0,60,220)
        _pill(frame, f"SCORE  {score}", (110,38), sc)
        _pill(frame, f"ERRORS  {errs}", (220,38), (0,60,220) if errs>0 else (255,255,255))
        for i, e in enumerate(stats.get("active_errors", [])):
            _banner(frame, e, i, w, h)
        if stats.get("lean_back"):
            _banner(frame, "LEAN BACK DETECTED", 0, w, h)

    elif exercise == "plank":
        score = stats.get("score", 100)
        dur   = stats.get("duration", 0)
        _pill(frame, f"HOLD  {int(dur)}s", (16, 38), (0,220,100))
        sc = (0,220,100) if score>=80 else (0,200,255) if score>=50 else (0,60,220)
        _pill(frame, f"QUALITY  {score}%", (110,38), sc)
        if stats.get("current_error"):
            _banner(frame, stats["current_error"], 0, w, h)


def _encode(frame, quality=80) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE — Bicep
# ═══════════════════════════════════════════════════════════════════════════

class _BicepSession:
    def __init__(self, gcn, scaler, device):
        self.gcn   = gcn;   self.scaler = scaler;  self.device = device
        self.left  = _ArmAnalyser("LEFT")
        self.right = _ArmAnalyser("RIGHT")
        self.posture    = "C"
        self.sl = self.sr = self.pl = self.pr = 0
        self.t0 = time.time()

    def process(self, lms, lm_names, mp_pose) -> dict:
        raw = {}
        for n in lm_names:
            lm = lms[mp_pose.PoseLandmark[n].value]
            c  = n.lower()
            raw[f"{c}_x"] = lm.x; raw[f"{c}_y"] = lm.y
            raw[f"{c}_z"] = lm.z; raw[f"{c}_v"] = lm.visibility

        pred, conf = _gcn_predict(raw, self.gcn, self.scaler, self.device)
        if conf >= 0.95: self.posture = pred
        lean = self.posture == "L"

        _, _, le = self.left.analyse(lms, lean)
        _, _, re = self.right.analyse(lms, lean)

        nl, nr = self.left.counter, self.right.counter
        if nl > self.pl:
            if not lean: self.sl += 5 if le else 10
            self.pl = nl
        if nr > self.pr:
            if not lean: self.sr += 5 if re else 10
            self.pr = nr

        total = nl + nr
        ts    = self.sl + self.sr
        mx    = total * 10 if total else 10
        pct   = int(ts / mx * 100) if mx else 0

        ae = []
        if self.left.loose_upper_arm  or self.right.loose_upper_arm:  ae.append("LOOSE UPPER ARM")
        if self.left.peak_angle != 1000 or self.right.peak_angle != 1000: ae.append("INCOMPLETE PEAK CONTRACTION")

        errs = (self.left.detected_errors["LOOSE_UPPER_ARM"]   + self.left.detected_errors["PEAK_CONTRACTION"] +
                self.right.detected_errors["LOOSE_UPPER_ARM"]  + self.right.detected_errors["PEAK_CONTRACTION"])

        return {"total_reps": total, "left_reps": nl, "right_reps": nr,
                "score": pct, "total_errors": errs, "active_errors": ae,
                "lean_back": lean, "duration": time.time() - self.t0}


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE — Plank
# ═══════════════════════════════════════════════════════════════════════════

class _PlankSession:
    def __init__(self, model, scaler):
        self.model   = model;  self.scaler  = scaler
        self.tracker = _PlankTracker()
        self.t0      = time.time()
        self._last_pred = "C"; self._last_conf = 0.0

    def process(self, results) -> dict:
        import pandas as pd
        row   = _extract_plank_keypoints(results)
        X     = self.scaler.transform(pd.DataFrame([row], columns=_PLANK_HEADERS))
        pred  = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        conf  = float(probs[np.argmax(probs)])
        self._last_pred = pred; self._last_conf = conf

        now = time.time()
        self.tracker.analyze(pred, conf, now)

        stats     = self.tracker.get_error_stats()
        err_sec   = stats["total_error_time"]
        if self.tracker.error_start_time:
            err_sec += now - self.tracker.error_start_time

        dur     = now - self.t0
        err_pct = (err_sec / dur * 100) if dur > 0 else 0
        quality = max(0, 100 - int(err_pct))

        return {"duration": dur, "score": quality,
                "low_back": stats["low_back_count"], "high_back": stats["high_back_count"],
                "current_error": self.tracker.current_error_label(pred, conf),
                "error_pct": round(err_pct, 1)}


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket — Bicep Curl
# ═══════════════════════════════════════════════════════════════════════════

@router.websocket("/api/live/bicep_curl")
async def live_bicep_curl(websocket: WebSocket):
    await websocket.accept()

    gcn_w = _ROOT / "bicep_curl_gcn.pth"
    gcn_s = _ROOT / "bicep_curl_gcn_scaler.pkl"
    if not gcn_w.exists() or not gcn_s.exists():
        await websocket.send_text(json.dumps({"error": f"GCN files not found in {_ROOT}"}))
        await websocket.close(); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn    = _PoseGCN(); gcn.load_state_dict(torch.load(gcn_w, map_location=device))
    gcn.to(device).eval()
    with open(gcn_s, "rb") as f: scaler = pickle.load(f)

    mp_pose  = mp.solutions.pose
    lm_names = ["NOSE","LEFT_SHOULDER","RIGHT_SHOULDER",
                "RIGHT_ELBOW","LEFT_ELBOW","RIGHT_WRIST","LEFT_WRIST",
                "LEFT_HIP","RIGHT_HIP"]
    session  = _BicepSession(gcn, scaler, device)

    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while True:
                data  = await websocket.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue
                res   = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                stats = {"total_reps":0,"left_reps":0,"right_reps":0,"score":0,
                         "total_errors":0,"active_errors":[],"lean_back":False,
                         "duration": time.time()-session.t0}
                if res.pose_landmarks:
                    _draw_skeleton(frame, res.pose_landmarks)
                    stats = session.process(res.pose_landmarks.landmark, lm_names, mp_pose)

                _hud(frame, stats, "bicep_curl")
                await websocket.send_text(json.dumps({"annotated_frame": _encode(frame), "stats": stats}))

    except WebSocketDisconnect: pass
    except Exception as e:
        try: await websocket.send_text(json.dumps({"error": str(e)}))
        except: pass


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket — Plank
# ═══════════════════════════════════════════════════════════════════════════

@router.websocket("/api/live/plank")
async def live_plank(websocket: WebSocket):
    await websocket.accept()

    mf = _ROOT / "plank_model.pkl"
    sf = _ROOT / "plank_input_scaler.pkl"
    if not mf.exists() or not sf.exists():
        await websocket.send_text(json.dumps({"error": f"Plank model files not found in {_ROOT}"}))
        await websocket.close(); return

    with open(mf, "rb") as f: model  = pickle.load(f)
    with open(sf, "rb") as f: scaler = pickle.load(f)

    mp_pose = mp.solutions.pose
    session = _PlankSession(model, scaler)

    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while True:
                data  = await websocket.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue
                res   = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                stats = {"duration": time.time()-session.t0, "score":100,
                         "low_back":0, "high_back":0, "current_error":None, "error_pct":0.0}
                if res.pose_landmarks:
                    _draw_skeleton(frame, res.pose_landmarks)
                    stats = session.process(res)

                _hud(frame, stats, "plank")
                await websocket.send_text(json.dumps({"annotated_frame": _encode(frame), "stats": stats}))

    except WebSocketDisconnect: pass
    except Exception as e:
        try: await websocket.send_text(json.dumps({"error": str(e)}))
        except: pass
        
        
        
# ═══════════════════════════════════════════════════════════════════════════
#  LUNGE + SQUAT — Live Webcam Detection
#  ─────────────────────────────────────────────────────────────────────────
#  Append everything below to the end of live_detector.py.
#  Nothing above (bicep/plank) is touched.
#
#  WebSocket endpoints added:
#    ws://host/api/live/lunge
#    ws://host/api/live/squat
#
#  Detection logic mirrors exercise_detector.py exactly:
#    Lunge  — two GCN models (stage n=3, error n=2), per-rep error flags
#    Squat  — one GCN model (stage n=3), transition-based foot/knee counting
#
#  All GCN classes, constants, normalisation functions are inlined here.
#  No imports from exercise_detector.py.
# ═══════════════════════════════════════════════════════════════════════════

import math as _math


# ───────────────────────────────────────────────────────────────────────────
# Shared angle helper (already exists in live_detector as _calculate_angle,
# but scoped locally to avoid any name clash)
# ───────────────────────────────────────────────────────────────────────────

def _live_angle(p1, p2, p3) -> float:
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = abs(rad * 180.0 / np.pi)
    return deg if deg <= 180 else 360 - deg


# ═══════════════════════════════════════════════════════════════════════════
#  ████  LUNGE  ████
# ═══════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────

_LL_LANDMARKS = [
    "nose",
    "left_shoulder",   "right_shoulder",
    "left_hip",        "right_hip",
    "left_knee",       "right_knee",
    "left_ankle",      "right_ankle",
    "left_heel",       "right_heel",
    "left_foot_index", "right_foot_index",
]
_LL_N_NODES = len(_LL_LANDMARKS)   # 13
_LL_N_FEATS = 4
_LL_LM_NAMES = [lm.upper() for lm in _LL_LANDMARKS]

_LL_EDGES = [
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("right_hip",      "right_knee"),
    ("left_knee",      "left_ankle"),
    ("right_knee",     "right_ankle"),
    ("left_hip",       "right_knee"),
    ("right_hip",      "left_knee"),
    ("left_ankle",     "left_heel"),
    ("right_ankle",    "right_heel"),
    ("left_ankle",     "left_foot_index"),
    ("right_ankle",    "right_foot_index"),
]
_ll_n2i = {n: i for i, n in enumerate(_LL_LANDMARKS)}
_ll_s, _ll_d = [], []
for _a, _b in _LL_EDGES:
    _i, _j = _ll_n2i[_a], _ll_n2i[_b]
    _ll_s += [_i, _j]; _ll_d += [_j, _i]
_LL_EDGE_INDEX = torch.tensor([_ll_s, _ll_d], dtype=torch.long)

_LL_FEAT_COLS = [f"{lm}_{c}" for lm in _LL_LANDMARKS for c in ["x", "y", "z", "v"]]


# ── GCN model (shared for stage n=3 and error n=2) ────────────────────────

class _LungeLiveGCN(nn.Module):
    def __init__(self, in_feats=4, hidden=64, out_feats=32, n_classes=3, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(out_feats)
        self.head  = nn.Sequential(
            nn.Linear(_LL_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = torch.nn.functional.relu(self.bn1(self.conv1(x, ei)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x, ei)))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x, ei)))
        x = x.view(batch.max().item() + 1, _LL_N_NODES * self.out_feats)
        return self.head(x)


# ── Normalisation ─────────────────────────────────────────────────────────

def _ll_normalize(row: dict) -> dict:
    cx = (row["left_shoulder_x"] + row["right_shoulder_x"] +
          row["left_hip_x"]      + row["right_hip_x"]) / 4
    cy = (row["left_shoulder_y"] + row["right_shoulder_y"] +
          row["left_hip_y"]      + row["right_hip_y"]) / 4
    smx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    smy = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hmx = (row["left_hip_x"]      + row["right_hip_x"])      / 2
    hmy = (row["left_hip_y"]      + row["right_hip_y"])      / 2
    torso = ((smx - hmx)**2 + (smy - hmy)**2)**0.5 + 1e-6
    out = dict(row)
    for lm in _LL_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - cx) / torso
        out[f"{lm}_y"] = (row[f"{lm}_y"] - cy) / torso
        out[f"{lm}_z"] =  row[f"{lm}_z"]        / torso
    return out


def _ll_predict(raw_row, model, scaler, device):
    """Returns (predicted_idx, confidence)."""
    norm     = _ll_normalize(raw_row)
    feat     = [[norm[c] for c in _LL_FEAT_COLS]]
    scaled   = scaler.transform(feat)
    x_node   = torch.tensor(scaled[0].reshape(_LL_N_NODES, _LL_N_FEATS), dtype=torch.float)
    data     = Data(x=x_node, edge_index=_LL_EDGE_INDEX)
    data.batch = torch.zeros(_LL_N_NODES, dtype=torch.long)
    data     = data.to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(data), dim=1)[0]
    idx  = int(probs.argmax().item())
    conf = float(probs[idx].item())
    return idx, conf


# ── Drawing helpers (lunge) ───────────────────────────────────────────────

_LL_BLACK  = (15,  15,  15)
_LL_WHITE  = (240, 240, 240)
_LL_ORANGE = (30,  120, 255)   # BGR
_LL_GREEN  = (60,  200,  60)
_LL_RED    = (40,   40, 220)
_LL_YELLOW = (0,   200, 255)
_LL_BLUE   = (200, 130,  40)

_LL_STAGE_META = {
    "init": ("STANDING",    _LL_BLUE),
    "mid":  ("DESCENT",     _LL_YELLOW),
    "down": ("FULL LUNGE",  _LL_GREEN),
    "":     ("---",         _LL_WHITE),
}


def _ll_draw_skeleton(frame, results, has_error: bool):
    import mediapipe as mp
    md   = mp.solutions.drawing_utils
    mpm  = mp.solutions.pose
    lc   = _LL_RED   if has_error else _LL_GREEN
    cc   = (50, 50, 210) if has_error else (50, 200, 50)
    md.draw_landmarks(
        frame, results.pose_landmarks, mpm.POSE_CONNECTIONS,
        md.DrawingSpec(color=lc, thickness=2, circle_radius=3),
        md.DrawingSpec(color=cc, thickness=2, circle_radius=1),
    )


def _ll_pill(frame, text: str, x: int, y: int, bg: tuple):
    import cv2
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, 0.44, 1)
    p = 6
    cv2.rectangle(frame, (x-p, y-th-p), (x+tw+p, y+bl+p), bg,      -1)
    cv2.rectangle(frame, (x-p, y-th-p), (x+tw+p, y+bl+p), _LL_BLACK, 1)
    cv2.putText(frame, text, (x, y), font, 0.44, _LL_BLACK, 1, cv2.LINE_AA)


def _ll_error_banner(frame, text: str, idx: int):
    import cv2
    h, w = frame.shape[:2]
    y = h - 36 - idx * 40
    ov = frame.copy()
    cv2.rectangle(ov, (0, y-26), (w, y+12), (25, 25, 200), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, f"!  {text}", (14, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, _LL_WHITE, 1, cv2.LINE_AA)


def _ll_draw_knee_angles(frame, landmarks, mp_pose,
                          right_angle, left_angle, right_err, left_err, w, h):
    import cv2
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    cv2.putText(frame, f"{int(right_angle)}",
                (int(rk.x*w)+14, int(rk.y*h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                _LL_RED if right_err else _LL_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{int(left_angle)}",
                (int(lk.x*w)+14, int(lk.y*h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                _LL_RED if left_err else _LL_GREEN, 2, cv2.LINE_AA)


def _ll_draw_hud(frame, counter, stage, score_pct,
                  kot_error, angle_error, rep_flash):
    import cv2
    h, w = frame.shape[:2]

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 70), _LL_BLACK, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, "LUNGE", (14, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, _LL_ORANGE, 1, cv2.LINE_AA)

    _ll_pill(frame, f"REPS  {counter}",   14,  56, _LL_GREEN)
    stage_txt, stage_col = _LL_STAGE_META.get(stage, ("---", _LL_WHITE))
    _ll_pill(frame, f"STAGE  {stage_txt}", 115, 56, stage_col)
    sc = _LL_GREEN if score_pct >= 80 else _LL_YELLOW if score_pct >= 50 else _LL_RED
    _ll_pill(frame, f"SCORE  {score_pct}%", 330, 56, sc)

    bi = 0
    if kot_error:
        _ll_error_banner(frame, "KNEE OVER TOE — STEP FURTHER FORWARD", bi); bi += 1
    if angle_error:
        _ll_error_banner(frame, "KNEE ANGLE OUT OF RANGE  [60-125 deg]", bi)

    # Rep flash (white + "REP!" centred)
    if rep_flash > 0:
        alpha = (rep_flash / 12.0) * 0.36
        fl = frame.copy()
        cv2.rectangle(fl, (0, 0), (w, h), _LL_WHITE, -1)
        cv2.addWeighted(fl, alpha, frame, 1 - alpha, 0, frame)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = "REP!"
        (tw, th), _ = cv2.getTextSize(label, font, 2.5, 3)
        cv2.putText(frame, label,
                    ((w-tw)//2, (h+th)//2),
                    font, 2.5, _LL_ORANGE, 3, cv2.LINE_AA)


# ── Lunge live session state ──────────────────────────────────────────────

class _LungeSession:
    """Mutable state for one live lunge session."""

    PROB_THRESHOLD    = 0.80
    KNEE_ANGLE_LO     = 60
    KNEE_ANGLE_HI     = 125

    _STAGE = {0: "D", 1: "I", 2: "M"}
    _ERR   = {0: "C", 1: "L"}

    def __init__(self, stage_model, sc_stage, err_model, sc_err, device):
        self.stage_model = stage_model
        self.sc_stage    = sc_stage
        self.err_model   = err_model
        self.sc_err      = sc_err
        self.device      = device

        self.current_stage       = ""
        self.counter             = 0
        self.score_total         = 0
        self.rep_flash_frames    = 0

        # Per-rep error flags (mirrors exercise_detector.py)
        self.rep_kot_seen   = False
        self.rep_angle_seen = False

        # Lifetime error totals
        self.kot_errors   = 0
        self.angle_errors = 0

        self.start_time = time.time()

    def process(self, landmarks, results, mp_pose, w, h):
        """
        Run one frame. Returns (stats_dict, annotated=False — caller draws).
        Actually: mutates state and returns the stats dict for the HUD.
        """
        # Decrement flash each call
        if self.rep_flash_frames > 0:
            self.rep_flash_frames -= 1

        # Build raw_row
        raw_row = {}
        for lm_name, lm_key in zip(_LL_LM_NAMES, _LL_LANDMARKS):
            lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
            raw_row[f"{lm_key}_x"] = lm_obj.x
            raw_row[f"{lm_key}_y"] = lm_obj.y
            raw_row[f"{lm_key}_z"] = lm_obj.z
            raw_row[f"{lm_key}_v"] = lm_obj.visibility

        # Stage prediction
        stage_idx, stage_conf = _ll_predict(raw_row, self.stage_model, self.sc_stage, self.device)
        stage_label = self._STAGE[stage_idx]

        if stage_conf >= self.PROB_THRESHOLD:
            if stage_label == "I":
                self.current_stage = "init"
            elif stage_label == "M":
                self.current_stage = "mid"
            elif stage_label == "D":
                if self.current_stage in ("init", "mid"):
                    self.counter += 1
                    self.rep_flash_frames = 12
                    # Score the completed rep
                    if self.rep_kot_seen:
                        pass               # unsafe rep → 0 pts
                    elif self.rep_angle_seen:
                        self.score_total += 5
                    else:
                        self.score_total += 10
                    self.rep_kot_seen   = False
                    self.rep_angle_seen = False
                self.current_stage = "down"

        # Error detection (only at "down")
        kot_error   = False
        angle_error = False
        right_angle = left_angle = 0.0
        right_err   = left_err  = False

        if self.current_stage == "down":
            err_idx, err_conf = _ll_predict(raw_row, self.err_model, self.sc_err, self.device)
            if self._ERR[err_idx] == "L" and err_conf >= self.PROB_THRESHOLD:
                kot_error = True

            # Count KOT once per rep
            if kot_error and not self.rep_kot_seen:
                self.kot_errors    += 1
                self.rep_kot_seen   = True

            # Knee angle (only if no KOT error)
            if not kot_error:
                def pt(name):
                    lm = landmarks[mp_pose.PoseLandmark[name].value]
                    return [lm.x, lm.y]
                right_angle = _live_angle(pt("RIGHT_HIP"), pt("RIGHT_KNEE"), pt("RIGHT_ANKLE"))
                left_angle  = _live_angle(pt("LEFT_HIP"),  pt("LEFT_KNEE"),  pt("LEFT_ANKLE"))
                right_err   = not (self.KNEE_ANGLE_LO <= right_angle <= self.KNEE_ANGLE_HI)
                left_err    = not (self.KNEE_ANGLE_LO <= left_angle  <= self.KNEE_ANGLE_HI)
                angle_error = right_err or left_err

                if angle_error and not self.rep_angle_seen:
                    self.angle_errors    += 1
                    self.rep_angle_seen   = True

        has_error  = kot_error or angle_error
        max_so_far = self.counter * 10 if self.counter > 0 else 10
        score_pct  = int((self.score_total / max_so_far) * 100)

        return {
            "has_error"   : has_error,
            "kot_error"   : kot_error,
            "angle_error" : angle_error,
            "right_angle" : right_angle,
            "left_angle"  : left_angle,
            "right_err"   : right_err,
            "left_err"    : left_err,
            "counter"     : self.counter,
            "current_stage": self.current_stage,
            "score_pct"   : score_pct,
            "rep_flash"   : self.rep_flash_frames,
            "duration"    : time.time() - self.start_time,
            "kot_errors"  : self.kot_errors,
            "angle_errors": self.angle_errors,
        }


# ── WebSocket endpoint: /api/live/lunge ───────────────────────────────────

@router.websocket("/api/live/lunge")
async def live_lunge_detection(websocket: WebSocket):
    """
    Live lunge detection over WebSocket.

    Client → Server : raw JPEG bytes
    Server → Client : JSON  { annotated_frame: <base64 JPEG>, stats: { … } }

    stats keys: counter, current_stage, score_pct, has_error,
                kot_error, angle_error, duration, kot_errors, angle_errors
    """
    await websocket.accept()

    stage_weights = _parent_dir / "lunge_stage_gcn.pth"
    stage_scaler  = _parent_dir / "lunge_stage_gcn_scaler.pkl"
    err_weights   = _parent_dir / "lunge_err_gcn.pth"
    err_scaler    = _parent_dir / "lunge_err_gcn_scaler.pkl"

    for p in (stage_weights, stage_scaler, err_weights, err_scaler):
        if not p.exists():
            await websocket.send_text(json.dumps(
                {"error": f"Lunge model file not found: {p}"}))
            await websocket.close()
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_model = _LungeLiveGCN(n_classes=3).to(device)
    stage_model.load_state_dict(torch.load(stage_weights, map_location=device))
    stage_model.eval()

    err_model = _LungeLiveGCN(n_classes=2).to(device)
    err_model.load_state_dict(torch.load(err_weights, map_location=device))
    err_model.eval()

    with open(stage_scaler, "rb") as f: sc_stage = pickle.load(f)
    with open(err_scaler,   "rb") as f: sc_err   = pickle.load(f)

    mp_pose = mp.solutions.pose
    session = _LungeSession(stage_model, sc_stage, err_model, sc_err, device)

    try:
        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7,
        ) as pose:
            while True:
                data = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                # Default stats when no pose detected
                stats = {
                    "counter": session.counter, "current_stage": session.current_stage,
                    "score_pct": 0, "has_error": False, "kot_error": False,
                    "angle_error": False, "duration": time.time() - session.start_time,
                    "kot_errors": session.kot_errors, "angle_errors": session.angle_errors,
                }

                if res.pose_landmarks:
                    landmarks = res.pose_landmarks.landmark
                    stats = session.process(landmarks, res, mp_pose, w, h)

                    # Draw skeleton
                    _ll_draw_skeleton(frame, res, stats["has_error"])

                    # Knee angles at DOWN and no KOT error
                    if session.current_stage == "down" and not stats["kot_error"]:
                        _ll_draw_knee_angles(
                            frame, landmarks, mp_pose,
                            stats["right_angle"], stats["left_angle"],
                            stats["right_err"],   stats["left_err"],
                            w, h
                        )

                # HUD (drawn even without pose — shows counters)
                _ll_draw_hud(
                    frame, stats["counter"], stats["current_stage"],
                    stats["score_pct"], stats["kot_error"], stats["angle_error"],
                    stats.get("rep_flash", 0)
                )

                b64 = _encode_frame(frame)
                await websocket.send_text(json.dumps({
                    "annotated_frame": b64,
                    "stats": {
                        "total_reps"  : stats["counter"],
                        "stage"       : stats["current_stage"],
                        "score"       : stats["score_pct"],
                        "has_error"   : stats["has_error"],
                        "kot_error"   : stats["kot_error"],
                        "angle_error" : stats["angle_error"],
                        "duration"    : round(stats["duration"], 1),
                        "kot_errors"  : stats["kot_errors"],
                        "angle_errors": stats["angle_errors"],
                    },
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  ████  SQUAT  ████
# ═══════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────

_SQ_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]
_SQ_N_NODES = len(_SQ_LANDMARKS)   # 9
_SQ_N_FEATS = 4
_SQ_LM_NAMES = [lm.upper() for lm in _SQ_LANDMARKS]

_SQ_EDGES = [
    ("nose",          "left_shoulder"),
    ("nose",          "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder","right_hip"),
    ("left_hip",      "right_hip"),
    ("left_hip",      "left_knee"),
    ("right_hip",     "right_knee"),
    ("left_knee",     "left_ankle"),
    ("right_knee",    "right_ankle"),
    ("left_hip",      "right_knee"),
    ("right_hip",     "left_knee"),
    ("left_ankle",    "right_ankle"),
]
_sq_n2i = {n: i for i, n in enumerate(_SQ_LANDMARKS)}
_sq_s, _sq_d = [], []
for _a, _b in _SQ_EDGES:
    _i, _j = _sq_n2i[_a], _sq_n2i[_b]
    _sq_s += [_i, _j]; _sq_d += [_j, _i]
_SQ_EDGE_INDEX = torch.tensor([_sq_s, _sq_d], dtype=torch.long)

_SQ_FEAT_COLS = [f"{lm}_{c}" for lm in _SQ_LANDMARKS for c in ["x", "y", "z", "v"]]

_SQ_FOOT_SHOULDER_THRESHOLDS = [1.2, 2.8]
_SQ_KNEE_FOOT_THRESHOLDS = {
    "up":     [0.5, 1.0],
    "middle": [0.7, 1.0],
    "down":   [0.7, 1.1],
}
_SQ_VISIBILITY_THRESHOLD = 0.6
_SQ_PROB_THRESHOLD       = 0.6


# ── GCN model ─────────────────────────────────────────────────────────────

class _SquatLiveGCN(nn.Module):
    def __init__(self, in_feats=4, hidden=64, out_feats=32, n_classes=3, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(out_feats)
        self.head  = nn.Sequential(
            nn.Linear(_SQ_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.dropout = dropout

    def forward(self, data):
        import torch.nn.functional as F
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, ei)))
        x = x.view(batch.max().item() + 1, _SQ_N_NODES * self.out_feats)
        return self.head(x)


# ── Normalisation ─────────────────────────────────────────────────────────

def _sq_normalize(row: dict) -> dict:
    cx = (row["left_shoulder_x"] + row["right_shoulder_x"] +
          row["left_hip_x"]      + row["right_hip_x"]) / 4
    cy = (row["left_shoulder_y"] + row["right_shoulder_y"] +
          row["left_hip_y"]      + row["right_hip_y"]) / 4
    smx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    smy = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hmx = (row["left_hip_x"]      + row["right_hip_x"])      / 2
    hmy = (row["left_hip_y"]      + row["right_hip_y"])      / 2
    torso = ((smx - hmx)**2 + (smy - hmy)**2)**0.5 + 1e-6
    out = dict(row)
    for lm in _SQ_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - cx) / torso
        out[f"{lm}_y"] = (row[f"{lm}_y"] - cy) / torso
        out[f"{lm}_z"] =  row[f"{lm}_z"]        / torso
    return out


def _sq_predict_stage(raw_row, model, scaler, label_encoder, device):
    """Returns (stage_str, confidence). 'unknown' if below threshold."""
    norm   = _sq_normalize(raw_row)
    feat   = [[norm[c] for c in _SQ_FEAT_COLS]]
    scaled = scaler.transform(feat)
    x_node = torch.tensor(scaled[0].reshape(_SQ_N_NODES, _SQ_N_FEATS), dtype=torch.float)
    data   = Data(x=x_node, edge_index=_SQ_EDGE_INDEX)
    data.batch = torch.zeros(_SQ_N_NODES, dtype=torch.long)
    data   = data.to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(data), dim=1)[0]
    idx  = int(probs.argmax().item())
    conf = float(probs[idx].item())
    if conf < _SQ_PROB_THRESHOLD:
        return "unknown", conf
    return label_encoder.inverse_transform([idx])[0], conf


def _sq_analyze_errors(landmarks, stage, mp_pose):
    """
    Geometric foot/knee placement analysis.
    Returns {"foot_placement": code, "knee_placement": code}
    Codes: -1=unknown, 0=correct, 1=too tight, 2=too wide
    Mirrors exercise_detector.py _analyze_squat_foot_knee_placement exactly.
    """
    result = {"foot_placement": -1, "knee_placement": -1}

    def vis(name): return landmarks[mp_pose.PoseLandmark[name].value].visibility
    def xy(name):
        p = landmarks[mp_pose.PoseLandmark[name].value]
        return [p.x, p.y]
    def dist(a, b): return _math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

    if any(vis(n) < _SQ_VISIBILITY_THRESHOLD for n in [
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_KNEE", "RIGHT_KNEE",
    ]):
        return result

    sw  = dist(xy("LEFT_SHOULDER"), xy("RIGHT_SHOULDER"))
    fw  = dist(xy("LEFT_FOOT_INDEX"), xy("RIGHT_FOOT_INDEX"))
    fsr = round(fw / sw, 1)
    lo, hi = _SQ_FOOT_SHOULDER_THRESHOLDS
    result["foot_placement"] = 0 if lo <= fsr <= hi else (1 if fsr < lo else 2)

    if result["foot_placement"] == 0:
        kw  = dist(xy("LEFT_KNEE"), xy("RIGHT_KNEE"))
        kfr = round(kw / fw, 1)
        thr = _SQ_KNEE_FOOT_THRESHOLDS.get(stage)
        if thr is not None:
            lo_k, hi_k = thr
            result["knee_placement"] = 0 if lo_k <= kfr <= hi_k else (1 if kfr < lo_k else 2)

    return result


# ── Drawing helpers (squat) ───────────────────────────────────────────────

_SQ_C_GREEN  = (0,   215, 95)
_SQ_C_RED    = (35,  40,  215)
_SQ_C_YELLOW = (0,   205, 255)
_SQ_C_WHITE  = (240, 240, 240)
_SQ_C_BLACK  = (12,  12,  12)
_SQ_C_ORANGE = (30,  130, 255)


def _sq_draw_skeleton(frame, results, has_error: bool):
    import mediapipe as mp
    md  = mp.solutions.drawing_utils
    mpm = mp.solutions.pose
    lc  = _SQ_C_RED   if has_error else _SQ_C_GREEN
    cc  = (50, 50, 210) if has_error else (60, 210, 60)
    md.draw_landmarks(
        frame, results.pose_landmarks, mpm.POSE_CONNECTIONS,
        md.DrawingSpec(color=lc, thickness=2, circle_radius=4),
        md.DrawingSpec(color=cc, thickness=2),
    )


def _sq_pill(frame, text: str, x: int, y: int, bg: tuple):
    import cv2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), bg,          -1)
    cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), _SQ_C_BLACK,  1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                _SQ_C_BLACK, 1, cv2.LINE_AA)


def _sq_error_banner(frame, text: str, idx: int):
    import cv2
    h, w = frame.shape[:2]
    y = h - 44 - idx * 38
    ov = frame.copy()
    cv2.rectangle(ov, (0, y-30), (w, y+10), (20, 20, 185), -1)
    cv2.addWeighted(ov, 0.76, frame, 0.24, 0, frame)
    cv2.putText(frame, f"!  {text}", (14, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, _SQ_C_WHITE, 1, cv2.LINE_AA)


def _sq_draw_hud(frame, stage, counter, score_pct, foot_label, knee_label, has_error):
    import cv2
    h, w = frame.shape[:2]

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 72), _SQ_C_BLACK, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, "SQUAT", (14, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, _SQ_C_ORANGE, 1, cv2.LINE_AA)

    stage_col = (_SQ_C_GREEN  if stage == "up"
                 else _SQ_C_YELLOW if stage == "middle"
                 else _SQ_C_RED    if stage == "down"
                 else (130, 130, 130))
    _sq_pill(frame, f"STAGE  {stage.upper()}", 14,  52, stage_col)
    _sq_pill(frame, f"REPS  {counter}",        155, 52, _SQ_C_GREEN)
    sc = _SQ_C_GREEN if score_pct >= 80 else _SQ_C_YELLOW if score_pct >= 50 else _SQ_C_RED
    _sq_pill(frame, f"SCORE  {score_pct}%",    255, 52, sc)

    bi = 0
    if foot_label in ("too tight", "too wide"):
        _sq_error_banner(frame, f"FOOT {foot_label.upper()}", bi); bi += 1
    if knee_label in ("too tight", "too wide"):
        _sq_error_banner(frame, f"KNEE {knee_label.upper()}", bi)

    # Red border on error
    if has_error:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), _SQ_C_RED, 3)


# ── Squat live session state ──────────────────────────────────────────────

class _SquatSession:
    """Mutable state for one live squat session. Mirrors exercise_detector.py."""

    _LABEL_MAP = {-1: "unknown", 0: "correct", 1: "too tight", 2: "too wide"}

    def __init__(self, gcn_model, scaler, label_encoder, device):
        self.model         = gcn_model
        self.scaler        = scaler
        self.label_encoder = label_encoder
        self.device        = device

        self.current_stage = "unknown"
        self.counter       = 0
        self.score_points  = 0

        # Per-rep error flags (reset at rep completion)
        self.rep_foot_err  = False
        self.rep_knee_err  = False

        # Transition tracking (the 190-error fix)
        self.previous_foot = "correct"
        self.previous_knee = "correct"

        # Lifetime error totals
        self.total_foot_errors = 0
        self.total_knee_errors = 0

        self.start_time = time.time()

    def process(self, landmarks, mp_pose):
        """Run one frame. Returns stats dict."""

        # Build raw_row
        raw_row = {}
        for lm_name in _SQ_LM_NAMES:
            lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
            col    = lm_name.lower()
            raw_row[f"{col}_x"] = lm_obj.x
            raw_row[f"{col}_y"] = lm_obj.y
            raw_row[f"{col}_z"] = lm_obj.z
            raw_row[f"{col}_v"] = lm_obj.visibility

        # Stage prediction
        predicted_stage, _ = _sq_predict_stage(
            raw_row, self.model, self.scaler, self.label_encoder, self.device
        )

        # Rep counter: down → up
        if predicted_stage == "down":
            self.current_stage = "down"
        elif self.current_stage == "down" and predicted_stage == "up":
            self.current_stage = "up"
            self.counter      += 1
            if self.rep_foot_err and self.rep_knee_err:
                self.score_points += 0
            elif self.rep_foot_err or self.rep_knee_err:
                self.score_points += 5
            else:
                self.score_points += 10
            self.rep_foot_err  = False
            self.rep_knee_err  = False
        elif predicted_stage not in ("unknown", ""):
            self.current_stage = predicted_stage

        # Geometric errors
        geo        = _sq_analyze_errors(landmarks, self.current_stage, mp_pose)
        foot_code  = geo["foot_placement"]
        knee_code  = geo["knee_placement"]
        foot_label = self._LABEL_MAP.get(foot_code, "unknown")
        knee_label = self._LABEL_MAP.get(knee_code, "unknown")

        # Transition-based counting
        if foot_label in ("too tight", "too wide"):
            if self.previous_foot != foot_label:
                self.total_foot_errors += 1
                self.rep_foot_err       = True
            self.previous_foot = foot_label
        else:
            self.previous_foot = foot_label

        if knee_label in ("too tight", "too wide"):
            if self.previous_knee != knee_label:
                self.total_knee_errors += 1
                self.rep_knee_err       = True
            self.previous_knee = knee_label
        else:
            self.previous_knee = knee_label

        has_error  = foot_label in ("too tight", "too wide") or \
                     knee_label in ("too tight", "too wide")
        max_s      = self.counter * 10 if self.counter > 0 else 10
        score_pct  = int((self.score_points / max_s) * 100) if max_s > 0 else 0

        return {
            "stage"            : self.current_stage,
            "counter"          : self.counter,
            "score_pct"        : score_pct,
            "foot_label"       : foot_label,
            "knee_label"       : knee_label,
            "has_error"        : has_error,
            "total_foot_errors": self.total_foot_errors,
            "total_knee_errors": self.total_knee_errors,
            "duration"         : time.time() - self.start_time,
        }


# ── WebSocket endpoint: /api/live/squat ──────────────────────────────────

@router.websocket("/api/live/squat")
async def live_squat_detection(websocket: WebSocket):
    """
    Live squat detection over WebSocket.

    Client → Server : raw JPEG bytes
    Server → Client : JSON  { annotated_frame: <base64 JPEG>, stats: { … } }

    stats keys: stage, counter, score_pct, foot_label, knee_label,
                has_error, total_foot_errors, total_knee_errors, duration
    """
    await websocket.accept()

    gcn_weights = _parent_dir / "squat_stage_gcn.pth"
    gcn_scaler  = _parent_dir / "squat_stage_gcn_scaler.pkl"
    gcn_le      = _parent_dir / "squat_stage_gcn_label_encoder.pkl"

    for p in (gcn_weights, gcn_scaler, gcn_le):
        if not p.exists():
            await websocket.send_text(json.dumps(
                {"error": f"Squat model file not found: {p}"}))
            await websocket.close()
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(gcn_scaler, "rb") as f: scaler        = pickle.load(f)
    with open(gcn_le,     "rb") as f: label_encoder = pickle.load(f)

    gcn_model = _SquatLiveGCN(n_classes=len(label_encoder.classes_)).to(device)
    gcn_model.load_state_dict(torch.load(gcn_weights, map_location=device))
    gcn_model.eval()

    mp_pose = mp.solutions.pose
    session = _SquatSession(gcn_model, scaler, label_encoder, device)

    try:
        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7,
        ) as pose:
            while True:
                data  = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                # Default stats
                stats = {
                    "stage"            : session.current_stage,
                    "counter"          : session.counter,
                    "score_pct"        : 0,
                    "foot_label"       : "unknown",
                    "knee_label"       : "unknown",
                    "has_error"        : False,
                    "total_foot_errors": session.total_foot_errors,
                    "total_knee_errors": session.total_knee_errors,
                    "duration"         : time.time() - session.start_time,
                }

                if res.pose_landmarks:
                    stats = session.process(res.pose_landmarks.landmark, mp_pose)
                    _sq_draw_skeleton(frame, res, stats["has_error"])

                # HUD always drawn
                _sq_draw_hud(
                    frame,
                    stats["stage"], stats["counter"], stats["score_pct"],
                    stats["foot_label"], stats["knee_label"], stats["has_error"]
                )

                b64 = _encode_frame(frame)
                await websocket.send_text(json.dumps({
                    "annotated_frame": b64,
                    "stats": {
                        "stage"            : stats["stage"],
                        "total_reps"       : stats["counter"],
                        "score"            : stats["score_pct"],
                        "foot_label"       : stats["foot_label"],
                        "knee_label"       : stats["knee_label"],
                        "has_error"        : stats["has_error"],
                        "total_foot_errors": stats["total_foot_errors"],
                        "total_knee_errors": stats["total_knee_errors"],
                        "duration"         : round(stats["duration"], 1),
                    },
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass