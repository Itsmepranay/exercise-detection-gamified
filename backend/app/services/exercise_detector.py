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
# Shared geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_angle(point1: list, point2: list, point3: list) -> float:
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    rad = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    deg = np.abs(rad * 180.0 / np.pi)
    return deg if deg <= 180 else 360 - deg


# ─────────────────────────────────────────────────────────────────────────────
# _Annotator — shared drawing primitives used by both detectors
#
# Design language:
#   • Colors: orange (#FF6B2B) accent, green correct, red error, yellow warning
#   • Barlow Condensed-style uppercase labels via cv2 FONT_HERSHEY_DUPLEX
#   • Semi-transparent overlays via addWeighted
#   • All coordinates are normalised → pixel via (w, h) multiply
# ─────────────────────────────────────────────────────────────────────────────

class _Annotator:
    # Brand palette
    ORANGE  = (43, 107, 255)   # BGR
    GREEN   = (80, 200, 80)
    RED     = (60,  60, 220)
    YELLOW  = (30, 180, 220)
    WHITE   = (240, 240, 240)
    BLACK   = (15,  15,  15)
    GRAY    = (100, 100, 100)
    DARK_BG = (28,  28,  28)

    # Skeleton connections — (landmark_a, landmark_b) using mp PoseLandmark names
    # Full body used for plank; upper body subset for bicep
    UPPER_BODY_CONNECTIONS = [
        ("NOSE",           "LEFT_SHOULDER"),
        ("NOSE",           "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER",  "LEFT_ELBOW"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("LEFT_ELBOW",     "LEFT_WRIST"),
        ("RIGHT_ELBOW",    "RIGHT_WRIST"),
        ("LEFT_SHOULDER",  "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP",       "RIGHT_HIP"),
    ]

    FULL_BODY_CONNECTIONS = UPPER_BODY_CONNECTIONS + [
        ("LEFT_HIP",       "LEFT_KNEE"),
        ("RIGHT_HIP",      "RIGHT_KNEE"),
        ("LEFT_KNEE",      "LEFT_ANKLE"),
        ("RIGHT_KNEE",     "RIGHT_ANKLE"),
        ("LEFT_ANKLE",     "LEFT_HEEL"),
        ("RIGHT_ANKLE",    "RIGHT_HEEL"),
        ("LEFT_ANKLE",     "LEFT_FOOT_INDEX"),
        ("RIGHT_ANKLE",    "RIGHT_FOOT_INDEX"),
    ]

    # Which joints to draw larger (active movement joints)
    ACTIVE_JOINTS = {
        "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_KNEE",  "RIGHT_KNEE",
    }

    @staticmethod
    def lm_px(landmark, w: int, h: int):
        """Convert normalised landmark to pixel coords."""
        return (int(landmark.x * w), int(landmark.y * h))

    @staticmethod
    def draw_skeleton(frame, landmarks, mp_pose, connections: list,
                      edge_color, node_color, error_nodes: set = None):
        """Draw skeleton edges then joints."""
        import cv2
        h, w = frame.shape[:2]
        error_nodes = error_nodes or set()

        # Edges
        for a_name, b_name in connections:
            try:
                a = landmarks[mp_pose.PoseLandmark[a_name].value]
                b = landmarks[mp_pose.PoseLandmark[b_name].value]
                if a.visibility < 0.4 or b.visibility < 0.4:
                    continue
                ax, ay = int(a.x * w), int(a.y * h)
                bx, by = int(b.x * w), int(b.y * h)
                # Thick outer stroke
                cv2.line(frame, (ax, ay), (bx, by), _Annotator.BLACK, 5, cv2.LINE_AA)
                cv2.line(frame, (ax, ay), (bx, by), edge_color, 2,  cv2.LINE_AA)
            except (KeyError, IndexError):
                pass

        # Joints
        drawn = set()
        for a_name, b_name in connections:
            for name in (a_name, b_name):
                if name in drawn:
                    continue
                drawn.add(name)
                try:
                    lm = landmarks[mp_pose.PoseLandmark[name].value]
                    if lm.visibility < 0.4:
                        continue
                    px, py = int(lm.x * w), int(lm.y * h)
                    r = 7 if name in _Annotator.ACTIVE_JOINTS else 5
                    col = _Annotator.RED if name in error_nodes else node_color
                    # Glow ring
                    cv2.circle(frame, (px, py), r + 4, (*col[:3], 60), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), r + 2, _Annotator.BLACK, 2,  cv2.LINE_AA)
                    cv2.circle(frame, (px, py), r,     col,              -1, cv2.LINE_AA)
                except (KeyError, IndexError):
                    pass

    @staticmethod
    def draw_hud_bar(frame, text_left: str, text_right: str, color_right=(240,240,240)):
        """Semi-transparent top HUD bar with left/right text."""
        import cv2
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), _Annotator.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        # Orange accent line
        cv2.rectangle(frame, (0, 40), (w, 43), _Annotator.ORANGE, -1)

        cv2.putText(frame, text_left.upper(),
                    (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                    _Annotator.ORANGE, 1, cv2.LINE_AA)
        cv2.putText(frame, text_right,
                    (w - 200, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                    color_right, 1, cv2.LINE_AA)

    @staticmethod
    def draw_pill(frame, x: int, y: int, text: str, bg_color, text_color=(15,15,15),
                  alpha=0.88):
        """Rounded pill badge."""
        import cv2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        pad_x, pad_y = 12, 7
        x1, y1 = x, y - th - pad_y
        x2, y2 = x + tw + pad_x * 2, y + pad_y
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text,
                    (x + pad_x, y), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    text_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_score_bar(frame, score_pct: float):
        """Thin score bar along the bottom — orange→green based on score."""
        import cv2
        h, w = frame.shape[:2]
        bar_h = 8
        y = h - bar_h
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y), (w, h), _Annotator.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        # Fill — interpolate orange→green
        fill_w = int(w * max(0, min(1, score_pct / 100)))
        t = score_pct / 100
        r = int((1 - t) * _Annotator.ORANGE[0] + t * _Annotator.GREEN[0])
        g = int((1 - t) * _Annotator.ORANGE[1] + t * _Annotator.GREEN[1])
        b = int((1 - t) * _Annotator.ORANGE[2] + t * _Annotator.GREEN[2])
        cv2.rectangle(frame, (0, y), (fill_w, h), (r, g, b), -1)

    @staticmethod
    def draw_angle_arc(frame, vertex, p1, p2, angle: float, color, radius=40):
        """Draw a protractor-style arc showing joint angle."""
        import cv2
        h, w = frame.shape[:2]
        vx, vy = int(vertex.x * w), int(vertex.y * h)
        p1x, p1y = int(p1.x * w), int(p1.y * h)

        # Angle of the limb from vertex toward p1
        start_ang = int(np.degrees(np.arctan2(p1y - vy, p1x - vx)))
        # Sweep the arc by the joint angle
        end_ang   = start_ang + int(angle)

        axes = (radius, radius)
        cv2.ellipse(frame, (vx, vy), axes, 0,
                    min(start_ang, end_ang), max(start_ang, end_ang),
                    _Annotator.BLACK, 4, cv2.LINE_AA)
        cv2.ellipse(frame, (vx, vy), axes, 0,
                    min(start_ang, end_ang), max(start_ang, end_ang),
                    color, 2, cv2.LINE_AA)
        # Angle text next to arc
        mid_ang = np.radians((start_ang + end_ang) / 2)
        tx = vx + int((radius + 12) * np.cos(mid_ang))
        ty = vy + int((radius + 12) * np.sin(mid_ang))
        cv2.putText(frame, f"{int(angle)}",
                    (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.45,
                    color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_body_angle_line(frame, landmarks, mp_pose,
                              shoulder_name, hip_name, ankle_name,
                              stage_color):
        """Draw the shoulder→hip→ankle alignment line for plank."""
        import cv2
        h, w = frame.shape[:2]
        try:
            sh  = landmarks[mp_pose.PoseLandmark[shoulder_name].value]
            hip = landmarks[mp_pose.PoseLandmark[hip_name].value]
            ank = landmarks[mp_pose.PoseLandmark[ankle_name].value]
            if sh.visibility < 0.4 or hip.visibility < 0.4 or ank.visibility < 0.4:
                return
            pts = [
                (int(sh.x*w),  int(sh.y*h)),
                (int(hip.x*w), int(hip.y*h)),
                (int(ank.x*w), int(ank.y*h)),
            ]
            # Outer glow
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], _Annotator.BLACK, 5, cv2.LINE_AA)
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], stage_color, 2, cv2.LINE_AA)

            # Hip angle marker dot
            cv2.circle(frame, pts[1], 8, _Annotator.BLACK, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[1], 6, stage_color,      -1, cv2.LINE_AA)
        except (KeyError, IndexError):
            pass

    @staticmethod
    def draw_error_bar(frame, error_pct: float):
        """Vertical red error time bar on the right edge for plank."""
        import cv2
        h, w = frame.shape[:2]
        bar_w = 10
        x = w - bar_w
        fill_h = int(h * max(0, min(1, error_pct / 100)))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, 0), (w, h), _Annotator.DARK_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if fill_h > 0:
            cv2.rectangle(frame, (x, h - fill_h), (w, h), _Annotator.RED, -1)

    @staticmethod
    def flash_red(frame, alpha=0.35):
        """Tint the whole frame red — used on error frames."""
        import cv2
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), _Annotator.RED, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def open_writer(video_path: str, output_path: str):
        """Create a cv2.VideoWriter matching the source video's fps and size."""
        import cv2
        cap  = cv2.VideoCapture(video_path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(output_path, fourcc, fps, (w, h)), fps, w, h


# ─────────────────────────────────────────────────────────────────────────────
# _BicepPoseAnalysis  (unchanged from previous version)
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
        has_error = False
        if not self._get_joints(landmarks):
            return (None, None, False)
        curl_angle = int(_calculate_angle(self.shoulder, self.elbow, self.wrist))
        if curl_angle > self.STAGE_DOWN_THRESHOLD:
            self.stage = "down"
        elif curl_angle < self.STAGE_UP_THRESHOLD and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        shoulder_projection = [self.shoulder[0], 1]
        ground_angle = int(_calculate_angle(self.elbow, self.shoulder, shoulder_projection))
        if lean_back_error:
            return (curl_angle, ground_angle, False)
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
            if (self.peak_contraction_angle != 1000
                    and self.peak_contraction_angle >= self.PEAK_CONTRACTION_THRESHOLD):
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
# GCN imports
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ─────────────────────────────────────────────────────────────────────────────
# Bicep GCN (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_BICEP_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "right_elbow",    "left_elbow",
    "right_wrist",    "left_wrist",
    "left_hip",       "right_hip",
]
_BICEP_N_NODES = len(_BICEP_LANDMARKS)
_BICEP_N_FEATS = 4

_BICEP_SKELETON_EDGES = [
    ("nose","left_shoulder"), ("nose","right_shoulder"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),  ("right_shoulder","right_elbow"),
    ("left_elbow","left_wrist"),     ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),    ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
]
_bicep_node_to_idx = {n: i for i, n in enumerate(_BICEP_LANDMARKS)}
_b_src, _b_dst = [], []
for _u, _v in _BICEP_SKELETON_EDGES:
    _i, _j = _bicep_node_to_idx[_u], _bicep_node_to_idx[_v]
    _b_src += [_i, _j]; _b_dst += [_j, _i]
_BICEP_EDGE_INDEX = torch.tensor([_b_src, _b_dst], dtype=torch.long)
_BICEP_FEATURE_COLS = [f"{lm}_{c}" for lm in _BICEP_LANDMARKS for c in ["x","y","z","v"]]


class _BicepPoseGCN(nn.Module):
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
            nn.Linear(_BICEP_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                          nn.ReLU(), nn.Dropout(dropout),
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
        x = x.view(batch.max().item() + 1, _BICEP_N_NODES * self.out_feats)
        return self.head(x)


def _normalize_bicep_pose(row: dict) -> dict:
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


def _predict_bicep_gcn(raw_row, gcn_model, scaler, device, threshold=0.95):
    norm        = _normalize_bicep_pose(raw_row)
    feat_vec    = [[norm[col] for col in _BICEP_FEATURE_COLS]]
    feat_scaled = scaler.transform(feat_vec)
    x_node = torch.tensor(feat_scaled[0].reshape(_BICEP_N_NODES, _BICEP_N_FEATS), dtype=torch.float)
    data        = Data(x=x_node, edge_index=_BICEP_EDGE_INDEX)
    data.batch  = torch.zeros(_BICEP_N_NODES, dtype=torch.long)
    data        = data.to(device)
    with torch.no_grad():
        logits = gcn_model(data)
        probs  = F.softmax(logits, dim=1)[0]
    prob_c, prob_l = probs[0].item(), probs[1].item()
    confidence      = max(prob_c, prob_l)
    predicted_class = "L" if prob_l >= (1 - threshold) else "C"
    return predicted_class, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Plank GCN (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_PLANK_LANDMARKS = [
    "nose",
    "left_shoulder",    "right_shoulder",
    "left_elbow",       "right_elbow",
    "left_wrist",       "right_wrist",
    "left_hip",         "right_hip",
    "left_knee",        "right_knee",
    "left_ankle",       "right_ankle",
    "left_heel",        "right_heel",
    "left_foot_index",  "right_foot_index",
]
_PLANK_N_NODES = len(_PLANK_LANDMARKS)
_PLANK_N_FEATS = 4

_PLANK_SKELETON_EDGES = [
    ("nose","left_shoulder"), ("nose","right_shoulder"),
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),  ("right_shoulder","right_elbow"),
    ("left_elbow","left_wrist"),     ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),    ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),        ("right_hip","right_knee"),
    ("left_knee","left_ankle"),      ("right_knee","right_ankle"),
    ("left_ankle","left_heel"),      ("right_ankle","right_heel"),
    ("left_ankle","left_foot_index"),("right_ankle","right_foot_index"),
]
_plank_node_to_idx = {n: i for i, n in enumerate(_PLANK_LANDMARKS)}
_p_src, _p_dst = [], []
for _u, _v in _PLANK_SKELETON_EDGES:
    _i, _j = _plank_node_to_idx[_u], _plank_node_to_idx[_v]
    _p_src += [_i, _j]; _p_dst += [_j, _i]
_PLANK_EDGE_INDEX   = torch.tensor([_p_src, _p_dst], dtype=torch.long)
_PLANK_FEATURE_COLS = [f"{lm}_{c}" for lm in _PLANK_LANDMARKS for c in ["x","y","z","v"]]
_PLANK_CLASS_C      = 0
_PLANK_CLASS_H      = 1
_PLANK_CLASS_L      = 2
_PLANK_CONFIDENCE_THRESHOLD = 0.6


class _PlankGCN(nn.Module):
    def __init__(self, in_feats=4, hidden=64, out_feats=32, n_classes=3, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(out_feats)
        self.head = nn.Sequential(
            nn.Linear(_PLANK_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                          nn.ReLU(), nn.Dropout(dropout),
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
        x = x.view(batch.max().item() + 1, _PLANK_N_NODES * self.out_feats)
        return self.head(x)


def _normalize_plank_pose(row: dict) -> dict:
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
    for lm in _PLANK_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - cx) / torso
        out[f"{lm}_y"] = (row[f"{lm}_y"] - cy) / torso
        out[f"{lm}_z"] =  row[f"{lm}_z"]        / torso
    return out


def _predict_plank_gcn(raw_row, gcn_model, scaler, device):
    norm        = _normalize_plank_pose(raw_row)
    feat_vec    = [[norm[col] for col in _PLANK_FEATURE_COLS]]
    feat_scaled = scaler.transform(feat_vec)
    x_node = torch.tensor(feat_scaled[0].reshape(_PLANK_N_NODES, _PLANK_N_FEATS), dtype=torch.float)
    data        = Data(x=x_node, edge_index=_PLANK_EDGE_INDEX)
    data.batch  = torch.zeros(_PLANK_N_NODES, dtype=torch.long)
    data        = data.to(device)
    with torch.no_grad():
        logits = gcn_model(data)
        probs  = F.softmax(logits, dim=1)[0]
    predicted_class_idx = int(probs.argmax().item())
    confidence          = float(probs[predicted_class_idx].item())
    return predicted_class_idx, confidence


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
# Annotations:
#   • Colored skeleton (green=correct, red=error, yellow=lean-back)
#   • Curl angle arc at both elbows
#   • Rep counter pill (top-left)
#   • Per-rep badge "GOOD REP ✓" / "FORM ERROR ✗" (top-right, fades after 60 frames)
#   • Error flash (red tint) for 3 frames when a bad rep is recorded
#   • Lean-back warning pill when GCN detects lean
#   • Live score bar (bottom)
#   • HUD bar (top) with exercise name + elapsed time
# ─────────────────────────────────────────────────────────────────────────────

class BicepCurlDetector(ExerciseDetector):
    POSTURE_ERROR_THRESHOLD = 0.95

    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import time
        import pickle

        # ── Load GCN ──────────────────────────────────────────────────────
        gcn_weights = parent_dir / "bicep_curl_gcn.pth"
        gcn_scaler  = parent_dir / "bicep_curl_gcn_scaler.pkl"
        if not gcn_weights.exists() or not gcn_scaler.exists():
            raise FileNotFoundError(
                f"GCN files not found in {parent_dir}. "
                "Expected: bicep_curl_gcn.pth and bicep_curl_gcn_scaler.pkl"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gcn_model = _BicepPoseGCN()
        gcn_model.load_state_dict(torch.load(gcn_weights, map_location=device))
        gcn_model.to(device).eval()
        with open(gcn_scaler, "rb") as f:
            scaler = pickle.load(f)

        mp_pose = mp.solutions.pose
        lm_names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
        ]

        left_arm  = _BicepPoseAnalysis("LEFT")
        right_arm = _BicepPoseAnalysis("RIGHT")

        stand_posture       = "C"
        score_left          = 0
        score_right         = 0
        prev_left_count     = 0
        prev_right_count    = 0
        left_rep_had_error  = False
        right_rep_had_error = False

        # ── Per-frame annotation state ────────────────────────────────────
        # Store per-frame data during detection pass, render in annotation pass
        frame_data = []   # list of dicts, one per frame that had landmarks

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_time = time.time()

        # ── PASS 1: detection ─────────────────────────────────────────────
        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.8, min_tracking_confidence=0.8,
        ) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if not res.pose_landmarks:
                    frame_data.append(None)
                    continue

                landmarks = res.pose_landmarks.landmark

                raw_row = {}
                for lm_name in lm_names:
                    lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
                    col    = lm_name.lower()
                    raw_row[f"{col}_x"] = lm_obj.x
                    raw_row[f"{col}_y"] = lm_obj.y
                    raw_row[f"{col}_z"] = lm_obj.z
                    raw_row[f"{col}_v"] = lm_obj.visibility

                predicted_posture, ml_confidence = _predict_bicep_gcn(
                    raw_row, gcn_model, scaler, device
                )
                if ml_confidence >= self.POSTURE_ERROR_THRESHOLD:
                    stand_posture = predicted_posture
                lean_back_error = (stand_posture == "L")

                curl_l, ground_l, left_err  = left_arm.analyze_pose(landmarks, lean_back_error)
                curl_r, ground_r, right_err = right_arm.analyze_pose(landmarks, lean_back_error)

                left_rep_had_error  |= left_err
                right_rep_had_error |= right_err

                rep_event  = None   # "good" | "bad" — fires when a rep completes this frame
                new_left   = left_arm.get_counter()
                new_right  = right_arm.get_counter()

                if new_left > prev_left_count:
                    if not lean_back_error:
                        score_left += 5 if left_rep_had_error else 10
                        rep_event = "bad" if left_rep_had_error else "good"
                    else:
                        rep_event = "bad"
                    left_rep_had_error = False
                    prev_left_count    = new_left

                if new_right > prev_right_count:
                    if not lean_back_error:
                        score_right += 5 if right_rep_had_error else 10
                        if rep_event != "bad":   # bad takes priority
                            rep_event = "bad" if right_rep_had_error else "good"
                    else:
                        rep_event = "bad"
                    right_rep_had_error = False
                    prev_right_count    = new_right

                total_reps_so_far = left_arm.get_counter() + right_arm.get_counter()
                total_score_so_far = score_left + score_right
                max_so_far = total_reps_so_far * 10 if total_reps_so_far > 0 else 10
                score_pct_so_far = int(total_score_so_far / max_so_far * 100) if max_so_far > 0 else 0

                frame_data.append({
                    "landmarks"     : res.pose_landmarks.landmark,
                    "lean_back"     : lean_back_error,
                    "has_error"     : (left_err or right_err) and not lean_back_error,
                    "rep_event"     : rep_event,       # fires the frame a rep completes
                    "total_reps"    : total_reps_so_far,
                    "score_pct"     : score_pct_so_far,
                    "curl_l"        : curl_l,
                    "curl_r"        : curl_r,
                    "elapsed"       : time.time() - start_time,
                })

        cap.release()
        duration = time.time() - start_time

        # ── Final stats ───────────────────────────────────────────────────
        total_reps   = left_arm.get_counter() + right_arm.get_counter()
        total_score  = score_left + score_right
        max_score    = total_reps * 10 if total_reps > 0 else 10
        score_pct    = int(total_score / max_score * 100) if max_score > 0 else 0
        total_errors = (
            left_arm.detected_errors["LOOSE_UPPER_ARM"]  +
            left_arm.detected_errors["PEAK_CONTRACTION"] +
            right_arm.detected_errors["LOOSE_UPPER_ARM"] +
            right_arm.detected_errors["PEAK_CONTRACTION"]
        )

        # ── PASS 2: annotated video ───────────────────────────────────────
        processed_video_filename = None
        if output_path:
            writer, _, w, h = _Annotator.open_writer(video_path, output_path)
            cap2 = cv2.VideoCapture(video_path)

            # Badge state: shown for 60 frames after a rep event
            badge_text      = ""
            badge_color     = _Annotator.GREEN
            badge_countdown = 0
            # Error flash: tint for 3 frames after a bad rep
            flash_countdown = 0

            for fd in frame_data:
                ret, frame = cap2.read()
                if not ret:
                    break

                if fd is None:
                    writer.write(frame)
                    continue

                lms         = fd["landmarks"]
                lean_back   = fd["lean_back"]
                has_error   = fd["has_error"]
                rep_event   = fd["rep_event"]
                total_reps  = fd["total_reps"]
                score_pct_f = fd["score_pct"]
                curl_l      = fd["curl_l"]
                curl_r      = fd["curl_r"]
                elapsed     = fd["elapsed"]

                # Choose skeleton color
                if lean_back:
                    edge_col = _Annotator.YELLOW
                    node_col = _Annotator.YELLOW
                elif has_error:
                    edge_col = _Annotator.RED
                    node_col = _Annotator.RED
                else:
                    edge_col = _Annotator.GREEN
                    node_col = _Annotator.GREEN

                # Error joints highlighted in red
                error_nodes = set()
                if has_error and not lean_back:
                    error_nodes = {"LEFT_ELBOW", "RIGHT_ELBOW"}

                # Draw skeleton
                _Annotator.draw_skeleton(
                    frame, lms, mp_pose,
                    _Annotator.UPPER_BODY_CONNECTIONS,
                    edge_col, node_col, error_nodes
                )

                # Curl angle arcs
                try:
                    le = lms[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    lw = lms[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    if curl_l is not None and le.visibility > 0.5:
                        arc_col = _Annotator.RED if has_error else _Annotator.ORANGE
                        _Annotator.draw_angle_arc(frame, le, ls, lw, curl_l, arc_col, radius=38)
                except Exception:
                    pass
                try:
                    re = lms[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    if curl_r is not None and re.visibility > 0.5:
                        arc_col = _Annotator.RED if has_error else _Annotator.ORANGE
                        _Annotator.draw_angle_arc(frame, re, rs, rw, curl_r, arc_col, radius=38)
                except Exception:
                    pass

                # Error flash
                if rep_event == "bad":
                    flash_countdown = 3
                if flash_countdown > 0:
                    _Annotator.flash_red(frame, alpha=0.28)
                    flash_countdown -= 1

                # Score bar (bottom)
                _Annotator.draw_score_bar(frame, score_pct_f)

                # HUD bar (top)
                mins, secs = divmod(int(elapsed), 60)
                time_str = f"{mins:02d}:{secs:02d}"
                _Annotator.draw_hud_bar(frame, "BICEP CURL", time_str)

                # Rep counter pill (top-left)
                _Annotator.draw_pill(
                    frame, 12, 80,
                    f"REPS  {total_reps}",
                    _Annotator.DARK_BG,
                    text_color=_Annotator.ORANGE
                )

                # Score pill (below reps)
                _Annotator.draw_pill(
                    frame, 12, 110,
                    f"SCORE  {score_pct_f}",
                    _Annotator.DARK_BG,
                    text_color=_Annotator.WHITE
                )

                # Lean-back warning
                if lean_back:
                    _Annotator.draw_pill(
                        frame, 12, 140,
                        "LEAN BACK  !",
                        _Annotator.YELLOW,
                        text_color=_Annotator.BLACK
                    )

                # Rep badge (top-right, fades)
                if rep_event == "good":
                    badge_text      = "GOOD REP  \u2713"
                    badge_color     = _Annotator.GREEN
                    badge_countdown = 60
                elif rep_event == "bad":
                    badge_text      = "FORM ERROR  \u2717"
                    badge_color     = _Annotator.RED
                    badge_countdown = 60

                if badge_countdown > 0:
                    alpha = min(0.9, badge_countdown / 60 * 0.9)
                    _Annotator.draw_pill(
                        frame, w - 200, 80,
                        badge_text, badge_color,
                        text_color=_Annotator.BLACK if badge_color == _Annotator.GREEN else _Annotator.WHITE,
                        alpha=alpha
                    )
                    badge_countdown -= 1

                writer.write(frame)

            cap2.release()
            writer.release()
            processed_video_filename = Path(output_path).name

        return {
            "score"           : score_pct,
            "duration_seconds": duration,
            "error_count"     : total_errors,
            "processed_video" : processed_video_filename,
            "details": {
                "left_reps"   : left_arm.get_counter(),
                "right_reps"  : right_arm.get_counter(),
                "total_reps"  : total_reps,
                "left_errors" : left_arm.detected_errors,
                "right_errors": right_arm.detected_errors,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# PlankDetector
# Annotations:
#   • Full-body skeleton (green=correct, red=error, yellow=high back)
#   • Shoulder→hip→ankle alignment line with stage color
#   • Stage pill top-left: "CORRECT ✓" / "HIGH BACK ⚠" / "LOW BACK ✗"
#   • Error time bar (right edge, fills red)
#   • Live quality score pill
#   • Error flash on stage transition into error
#   • HUD bar (top) with exercise name + elapsed time
#   • Score bar (bottom)
# ─────────────────────────────────────────────────────────────────────────────

class PlankDetector(ExerciseDetector):
    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import time
        import pickle

        gcn_weights     = parent_dir / "plank_gcn.pth"
        gcn_scaler_file = parent_dir / "plank_gcn_scaler.pkl"
        if not gcn_weights.exists() or not gcn_scaler_file.exists():
            raise FileNotFoundError(
                f"Plank GCN files not found in {parent_dir}. "
                "Expected: plank_gcn.pth and plank_gcn_scaler.pkl"
            )

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gcn_model = _PlankGCN()
        gcn_model.load_state_dict(torch.load(gcn_weights, map_location=device))
        gcn_model.to(device).eval()
        with open(gcn_scaler_file, "rb") as f:
            scaler = pickle.load(f)

        mp_pose  = mp.solutions.pose
        lm_names = [lm.upper() for lm in _PLANK_LANDMARKS]

        previous_stage   = "unknown"
        has_error        = False
        error_start_time = None
        total_error_time = 0.0
        low_back_count   = 0
        high_back_count  = 0

        frame_data = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        session_start_time = time.time()

        # ── PASS 1: detection ─────────────────────────────────────────────
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

                if not res.pose_landmarks:
                    frame_data.append(None)
                    continue

                raw_row = {}
                for lm_name in lm_names:
                    lm_obj = res.pose_landmarks.landmark[
                        mp_pose.PoseLandmark[lm_name].value
                    ]
                    col = lm_name.lower()
                    raw_row[f"{col}_x"] = lm_obj.x
                    raw_row[f"{col}_y"] = lm_obj.y
                    raw_row[f"{col}_z"] = lm_obj.z
                    raw_row[f"{col}_v"] = lm_obj.visibility

                predicted_class_idx, confidence = _predict_plank_gcn(
                    raw_row, gcn_model, scaler, device
                )

                if confidence >= _PLANK_CONFIDENCE_THRESHOLD:
                    if predicted_class_idx == _PLANK_CLASS_C:
                        current_stage = "correct"
                    elif predicted_class_idx == _PLANK_CLASS_H:
                        current_stage = "high back"
                    else:
                        current_stage = "low back"
                else:
                    current_stage = "unknown"

                stage_transition = False   # True when entering error this frame
                if current_stage in ("low back", "high back"):
                    if previous_stage != current_stage:
                        if current_stage == "low back":
                            low_back_count += 1
                        else:
                            high_back_count += 1
                        if not has_error:
                            error_start_time = current_time
                        has_error        = True
                        stage_transition = True
                    # else: continuing same error, no action
                else:
                    if has_error and error_start_time is not None:
                        total_error_time += current_time - error_start_time
                        error_start_time  = None
                    has_error = False

                previous_stage = current_stage

                total_time_so_far = current_time - session_start_time
                err_pct_so_far    = (total_error_time / total_time_so_far * 100
                                     if total_time_so_far > 0 else 0)
                quality_so_far    = max(0, 100 - int(err_pct_so_far))

                frame_data.append({
                    "landmarks"        : res.pose_landmarks.landmark,
                    "stage"            : current_stage,
                    "has_error"        : has_error,
                    "stage_transition" : stage_transition,
                    "error_pct"        : err_pct_so_far,
                    "quality_score"    : quality_so_far,
                    "elapsed"          : current_time - session_start_time,
                })

        cap.release()
        total_time = time.time() - session_start_time

        if has_error and error_start_time is not None:
            total_error_time += time.time() - error_start_time

        error_pct     = (total_error_time / total_time * 100) if total_time > 0 else 0
        quality_score = max(0, 100 - int(error_pct))
        total_errors  = low_back_count + high_back_count

        # ── PASS 2: annotated video ───────────────────────────────────────
        processed_video_filename = None
        if output_path:
            writer, _, w, h = _Annotator.open_writer(video_path, output_path)
            cap2 = cv2.VideoCapture(video_path)

            flash_countdown = 0

            for fd in frame_data:
                ret, frame = cap2.read()
                if not ret:
                    break

                if fd is None:
                    writer.write(frame)
                    continue

                lms             = fd["landmarks"]
                stage           = fd["stage"]
                has_err         = fd["has_error"]
                transition      = fd["stage_transition"]
                err_pct_f       = fd["error_pct"]
                quality_f       = fd["quality_score"]
                elapsed         = fd["elapsed"]

                # Stage → colors
                if stage == "correct":
                    edge_col  = _Annotator.GREEN
                    node_col  = _Annotator.GREEN
                    stage_col = _Annotator.GREEN
                    pill_txt  = "CORRECT  \u2713"
                    pill_bg   = _Annotator.GREEN
                    pill_tc   = _Annotator.BLACK
                elif stage == "high back":
                    edge_col  = _Annotator.YELLOW
                    node_col  = _Annotator.YELLOW
                    stage_col = _Annotator.YELLOW
                    pill_txt  = "HIGH BACK  \u26a0"
                    pill_bg   = _Annotator.YELLOW
                    pill_tc   = _Annotator.BLACK
                elif stage == "low back":
                    edge_col  = _Annotator.RED
                    node_col  = _Annotator.RED
                    stage_col = _Annotator.RED
                    pill_txt  = "LOW BACK  \u2717"
                    pill_bg   = _Annotator.RED
                    pill_tc   = _Annotator.WHITE
                else:
                    edge_col  = _Annotator.GRAY
                    node_col  = _Annotator.GRAY
                    stage_col = _Annotator.GRAY
                    pill_txt  = "DETECTING..."
                    pill_bg   = _Annotator.DARK_BG
                    pill_tc   = _Annotator.WHITE

                # Error nodes for plank: hips highlighted when back is wrong
                error_nodes = set()
                if stage in ("high back", "low back"):
                    error_nodes = {"LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"}

                # Draw full skeleton
                _Annotator.draw_skeleton(
                    frame, lms, mp_pose,
                    _Annotator.FULL_BODY_CONNECTIONS,
                    edge_col, node_col, error_nodes
                )

                # Body alignment line (left side — usually most visible in plank)
                _Annotator.draw_body_angle_line(
                    frame, lms, mp_pose,
                    "LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE",
                    stage_col
                )

                # Error flash on transition into error
                if transition:
                    flash_countdown = 4
                if flash_countdown > 0:
                    _Annotator.flash_red(frame, alpha=0.30)
                    flash_countdown -= 1

                # Error time bar (right edge)
                _Annotator.draw_error_bar(frame, err_pct_f)

                # Score bar (bottom)
                _Annotator.draw_score_bar(frame, quality_f)

                # HUD bar (top)
                mins, secs = divmod(int(elapsed), 60)
                time_str = f"{mins:02d}:{secs:02d}"
                _Annotator.draw_hud_bar(frame, "PLANK", time_str)

                # Stage pill (top-left)
                _Annotator.draw_pill(frame, 12, 80, pill_txt, pill_bg, text_color=pill_tc)

                # Quality score pill (below stage)
                _Annotator.draw_pill(
                    frame, 12, 110,
                    f"QUALITY  {quality_f}%",
                    _Annotator.DARK_BG,
                    text_color=_Annotator.WHITE
                )

                writer.write(frame)

            cap2.release()
            writer.release()
            processed_video_filename = Path(output_path).name

        return {
            "score"           : quality_score,
            "duration_seconds": total_time,
            "error_count"     : total_errors,
            "processed_video" : processed_video_filename,
            "details": {
                "low_back_errors"   : low_back_count,
                "high_back_errors"  : high_back_count,
                "error_time_seconds": total_error_time,
                "error_percentage"  : error_pct,
            },
        }
        
        
# ─────────────────────────────────────────────────────────────────────────────
# LungeGCN — shared model class for both stage and error detectors
#
# Architecture: 13 nodes × 4 features → 3 GCNConv layers → flatten → FC head
#   Stage model : n_classes=3  (D=0, I=1, M=2  after LabelEncoder)
#   Error model : n_classes=2  (C=0, L=1  after LabelEncoder)
#
# Flatten head input: 13 nodes × 32 out_feats = 416
# Must match the training notebook (lunge_gcn_training.ipynb) exactly.
# ─────────────────────────────────────────────────────────────────────────────

_LUNGE_LANDMARKS = [
    "nose",
    "left_shoulder",    "right_shoulder",
    "left_hip",         "right_hip",
    "left_knee",        "right_knee",
    "left_ankle",       "right_ankle",
    "left_heel",        "right_heel",
    "left_foot_index",  "right_foot_index",
]
_LUNGE_N_NODES = len(_LUNGE_LANDMARKS)   # 13
_LUNGE_N_FEATS = 4                        # x, y, z, visibility

_LUNGE_SKELETON_EDGES = [
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
    # Cross-knee edges — capture asymmetric lunge stance
    ("left_hip",       "right_knee"),
    ("right_hip",      "left_knee"),
    # Feet
    ("left_ankle",     "left_heel"),
    ("right_ankle",    "right_heel"),
    ("left_ankle",     "left_foot_index"),
    ("right_ankle",    "right_foot_index"),
]

_lunge_node_to_idx = {n: i for i, n in enumerate(_LUNGE_LANDMARKS)}
_lunge_src, _lunge_dst = [], []
for _lu, _lv in _LUNGE_SKELETON_EDGES:
    _li, _lj = _lunge_node_to_idx[_lu], _lunge_node_to_idx[_lv]
    _lunge_src += [_li, _lj]
    _lunge_dst += [_lj, _li]
_LUNGE_EDGE_INDEX = torch.tensor([_lunge_src, _lunge_dst], dtype=torch.long)

# Feature column order — must match the scaler fitted during training
_LUNGE_FEATURE_COLS = [
    f"{lm}_{coord}"
    for lm in _LUNGE_LANDMARKS
    for coord in ["x", "y", "z", "v"]
]

# MediaPipe landmark names in the same order as _LUNGE_LANDMARKS
_LUNGE_LM_NAMES = [
    "NOSE",
    "LEFT_SHOULDER",    "RIGHT_SHOULDER",
    "LEFT_HIP",         "RIGHT_HIP",
    "LEFT_KNEE",        "RIGHT_KNEE",
    "LEFT_ANKLE",       "RIGHT_ANKLE",
    "LEFT_HEEL",        "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",  "RIGHT_FOOT_INDEX",
]


class _LungeGCN(nn.Module):
    """
    3-layer GCNConv → flatten → FC head.
    Reused for both stage (n_classes=3) and error (n_classes=2) models.
    """
    def __init__(self, in_feats=4, hidden=64, out_feats=32,
                 n_classes=3, dropout=0.4):
        super().__init__()
        self.out_feats = out_feats

        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,   hidden)
        self.conv3 = GCNConv(hidden,   out_feats)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(out_feats)

        # 13 × 32 = 416
        self.head = nn.Sequential(
            nn.Linear(_LUNGE_N_NODES * out_feats, 128),
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
        x = x.view(batch_size, _LUNGE_N_NODES * self.out_feats)
        return self.head(x)


def _normalize_lunge_row(row: dict) -> dict:
    """
    Camera-invariant normalisation — identical to the training notebook:
      1. Subtract torso centre  (shoulder + hip midpoint average)
      2. Divide by torso size   (shoulder-mid to hip-mid distance)
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
    for lm in _LUNGE_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - centre_x) / torso_size
        out[f"{lm}_y"] = (row[f"{lm}_y"] - centre_y) / torso_size
        out[f"{lm}_z"] =  row[f"{lm}_z"]              / torso_size
        # visibility unchanged
    return out


def _predict_lunge_gcn(raw_row: dict, gcn_model: _LungeGCN,
                        scaler, device: torch.device) -> tuple:
    """
    Run one frame through a LungeGCN model.

    Returns
    -------
    predicted_idx : int   — raw class index from argmax
    probs         : list  — softmax probabilities for all classes
    confidence    : float — max probability
    """
    norm        = _normalize_lunge_row(raw_row)
    feat_vec    = [[norm[col] for col in _LUNGE_FEATURE_COLS]]
    feat_scaled = scaler.transform(feat_vec)

    x_node = torch.tensor(
        feat_scaled[0].reshape(_LUNGE_N_NODES, _LUNGE_N_FEATS),
        dtype=torch.float
    )
    data       = Data(x=x_node, edge_index=_LUNGE_EDGE_INDEX)
    data.batch = torch.zeros(_LUNGE_N_NODES, dtype=torch.long)
    data       = data.to(device)

    with torch.no_grad():
        logits = gcn_model(data)              # [1, n_classes]
        probs  = F.softmax(logits, dim=1)[0]  # [n_classes]

    probs_list    = probs.cpu().numpy().tolist()
    predicted_idx = int(probs.argmax().item())
    confidence    = float(probs[predicted_idx].item())

    return predicted_idx, probs_list, confidence


# ─────────────────────────────────────────────────────────────────────────────
# LungeDetector
# ─────────────────────────────────────────────────────────────────────────────

class LungeDetector(ExerciseDetector):
    """
    Lunge exercise detector.

    Stage detection  : _LungeGCN (n_classes=3)  → I / M / D
                       lunge_stage_gcn.pth  +  lunge_stage_gcn_scaler.pkl

    Error detection  : _LungeGCN (n_classes=2)  → C / L  (knee-over-toe)
                       lunge_err_gcn.pth    +  lunge_err_gcn_scaler.pkl
                       runs only at the DOWN stage (same as original lunge.py)

    Knee angle check : geometric rule on raw landmarks, also only at DOWN stage.
                       Thresholds: [60°, 125°]  (same as original lunge.py)

    Rep counting     : I → M → D transition completes one rep
                       (current_stage must be "init" or "mid" before hitting "down")

    Scoring          : 10 pts per clean rep, 5 pts if only knee-angle error,
                       0 pts if knee-over-toe (ML) error is active at rep completion.
    """

    PREDICTION_PROB_THRESHOLD = 0.80   # same as original lunge.py
    KNEE_ANGLE_THRESHOLD      = [60, 125]

    # Stage label mapping — LabelEncoder sorts alphabetically:
    #   D → 0,  I → 1,  M → 2
    _STAGE_IDX = {"D": 0, "I": 1, "M": 2}
    _STAGE_LABEL = {0: "D", 1: "I", 2: "M"}

    # Error label mapping — LabelEncoder sorts alphabetically:
    #   C → 0,  L → 1
    _ERR_IDX   = {"C": 0, "L": 1}
    _ERR_LABEL = {0: "C", 1: "L"}

    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import time
        import pickle

        # ── Load stage GCN ────────────────────────────────────────────────
        stage_weights = parent_dir / "lunge_stage_gcn.pth"
        stage_scaler  = parent_dir / "lunge_stage_gcn_scaler.pkl"
        if not stage_weights.exists() or not stage_scaler.exists():
            raise FileNotFoundError(
                f"Lunge stage GCN files not found in {parent_dir}. "
                "Expected: lunge_stage_gcn.pth  and  lunge_stage_gcn_scaler.pkl"
            )

        # ── Load error GCN ────────────────────────────────────────────────
        err_weights = parent_dir / "lunge_err_gcn.pth"
        err_scaler  = parent_dir / "lunge_err_gcn_scaler.pkl"
        if not err_weights.exists() or not err_scaler.exists():
            raise FileNotFoundError(
                f"Lunge error GCN files not found in {parent_dir}. "
                "Expected: lunge_err_gcn.pth  and  lunge_err_gcn_scaler.pkl"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        stage_model = _LungeGCN(n_classes=3).to(device)
        stage_model.load_state_dict(torch.load(stage_weights, map_location=device))
        stage_model.eval()

        err_model = _LungeGCN(n_classes=2).to(device)
        err_model.load_state_dict(torch.load(err_weights, map_location=device))
        err_model.eval()

        with open(stage_scaler, "rb") as f:
            sc_stage = pickle.load(f)
        with open(err_scaler, "rb") as f:
            sc_err = pickle.load(f)

        mp_pose = mp.solutions.pose

        # ── Session state ─────────────────────────────────────────────────
        current_stage  = ""   # "init" | "mid" | "down"
        counter        = 0

        # Error tracking
        knee_over_toe_errors = 0
        knee_angle_errors    = 0

        # Scoring
        score_total = 0
        prev_counter = 0

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

                # ── Build raw landmark dict ───────────────────────────────
                raw_row = {}
                for lm_name, lm_key in zip(_LUNGE_LM_NAMES, _LUNGE_LANDMARKS):
                    lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
                    raw_row[f"{lm_key}_x"] = lm_obj.x
                    raw_row[f"{lm_key}_y"] = lm_obj.y
                    raw_row[f"{lm_key}_z"] = lm_obj.z
                    raw_row[f"{lm_key}_v"] = lm_obj.visibility

                # ── Stage GCN prediction ──────────────────────────────────
                stage_idx, stage_probs, stage_conf = _predict_lunge_gcn(
                    raw_row, stage_model, sc_stage, device
                )
                stage_label = self._STAGE_LABEL[stage_idx]

                # Update stage only when confidence clears the threshold
                # — same probability-gating as original lunge.py
                if stage_conf >= self.PREDICTION_PROB_THRESHOLD:
                    if stage_label == "I":
                        current_stage = "init"
                    elif stage_label == "M":
                        current_stage = "mid"
                    elif stage_label == "D":
                        # Rep completes the moment we hit DOWN from init or mid
                        if current_stage in ("init", "mid"):
                            counter += 1
                        current_stage = "down"

                # ── Error GCN + knee-angle check (DOWN stage only) ────────
                # Mirrors original lunge.py: both checks only at "down"
                knee_over_toe_error = False
                knee_angle_error    = False

                if current_stage == "down":

                    # ML error check (knee-over-toe)
                    err_idx, err_probs, err_conf = _predict_lunge_gcn(
                        raw_row, err_model, sc_err, device
                    )
                    err_label = self._ERR_LABEL[err_idx]

                    if (err_label == "L" and
                            err_conf >= self.PREDICTION_PROB_THRESHOLD):
                        knee_over_toe_error = True

                    elif (err_label == "C" and
                            err_conf >= self.PREDICTION_PROB_THRESHOLD):
                        knee_over_toe_error = False

                    # Geometric knee-angle check
                    # Only evaluated when knee-over-toe is NOT active —
                    # matches original lunge.py analyze_knee_angle logic
                    if not knee_over_toe_error:
                        right_hip   = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        right_knee  = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        left_hip    = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee   = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_ankle  = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        right_angle = _calculate_angle(right_hip, right_knee, right_ankle)
                        left_angle  = _calculate_angle(left_hip,  left_knee,  left_ankle)

                        lo, hi = self.KNEE_ANGLE_THRESHOLD
                        right_angle_error = not (lo <= right_angle <= hi)
                        left_angle_error  = not (lo <= left_angle  <= hi)
                        knee_angle_error  = right_angle_error or left_angle_error

                # ── Scoring on rep completion ─────────────────────────────
                # Evaluate at the moment the counter increments (same delta
                # pattern used in BicepCurlDetector)
                if counter > prev_counter:
                    if knee_over_toe_error:
                        # Bad rep — no points (mirrors original: 0 score for KOT error)
                        pass
                    elif knee_angle_error:
                        score_total += 5    # partial credit
                        knee_angle_errors += 1
                    else:
                        score_total += 10   # clean rep
                    prev_counter = counter

                # Track cumulative error counts
                if knee_over_toe_error:
                    knee_over_toe_errors += 1

        cap.release()
        duration = time.time() - start_time

        max_score   = counter * 10 if counter > 0 else 10
        score_pct   = int((score_total / max_score) * 100) if max_score > 0 else 0
        total_errors = knee_over_toe_errors + knee_angle_errors

        return {
            "score"           : score_pct,
            "duration_seconds": duration,
            "error_count"     : total_errors,
            "details": {
                "total_reps"          : counter,
                "knee_over_toe_errors": knee_over_toe_errors,
                "knee_angle_errors"   : knee_angle_errors,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_DETECTORS: Dict[str, ExerciseDetector] = {
    "bicep_curl": BicepCurlDetector(),
    "plank"     : PlankDetector(),
    "lunge"     : LungeDetector(),
}

def get_detector(exercise_name: str) -> ExerciseDetector:
    if exercise_name not in EXERCISE_DETECTORS:
        raise ValueError(f"No detector found for exercise: {exercise_name}")
    return EXERCISE_DETECTORS[exercise_name]