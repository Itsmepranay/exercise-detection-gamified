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
    ("left_hip",       "right_knee"),
    ("right_hip",      "left_knee"),
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

_LUNGE_FEATURE_COLS = [
    f"{lm}_{coord}"
    for lm in _LUNGE_LANDMARKS
    for coord in ["x", "y", "z", "v"]
]

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
    return out


def _predict_lunge_gcn(raw_row: dict, gcn_model: _LungeGCN,
                        scaler, device: torch.device) -> tuple:
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
        logits = gcn_model(data)
        probs  = F.softmax(logits, dim=1)[0]
    probs_list    = probs.cpu().numpy().tolist()
    predicted_idx = int(probs.argmax().item())
    confidence    = float(probs[predicted_idx].item())
    return predicted_idx, probs_list, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Lunge annotation helpers
# All colours in BGR (OpenCV convention)
# ─────────────────────────────────────────────────────────────────────────────

_L_BLACK  = (15,  15,  15)
_L_WHITE  = (245, 245, 240)
_L_ORANGE = (43,  107, 255)   # BGR for RepTrack orange #FF6B2B
_L_GREEN  = (80,  210,  60)   # correct form
_L_RED    = (50,   55, 220)   # error / warning
_L_YELLOW = (0,   205, 255)   # mid stage / caution
_L_BLUE   = (210, 140,  50)   # init stage

# Stage label → (display string, pill colour)
_STAGE_DISPLAY = {
    "init": ("STANDING",   _L_BLUE),
    "mid" : ("DESCENT",    _L_YELLOW),
    "down": ("FULL LUNGE", _L_GREEN),
    ""    : ("---",        _L_WHITE),
}


def _lunge_draw_skeleton(frame, mp_results, has_error: bool):
    """Full-body skeleton; green when correct, red on any error."""
    import mediapipe as mp
    mp_drawing  = mp.solutions.drawing_utils
    mp_pose_mod = mp.solutions.pose

    lm_col   = _L_RED  if has_error else _L_GREEN
    conn_col = (60, 60, 200) if has_error else (60, 200, 60)

    mp_drawing.draw_landmarks(
        frame,
        mp_results.pose_landmarks,
        mp_pose_mod.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=lm_col,  thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=conn_col, thickness=2, circle_radius=1),
    )


def _lunge_pill(frame, text: str, pos: tuple, bg_color: tuple,
                font_scale: float = 0.44, thickness: int = 1):
    """Filled rounded-rect label pill."""
    import cv2
    x, y = pos
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad),
                  bg_color, -1)
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad),
                  _L_BLACK, 1)
    cv2.putText(frame, text, (x, y), font, font_scale,
                _L_BLACK, thickness, cv2.LINE_AA)


def _lunge_error_banner(frame, text: str, idx: int, w: int, h: int):
    """Semi-transparent red strip at the bottom with warning text."""
    import cv2
    y = h - 36 - idx * 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 26), (w, y + 12), (28, 28, 195), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, f"!  {text}", (14, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.47, _L_WHITE, 1, cv2.LINE_AA)


def _lunge_draw_knee_angles(frame, landmarks, mp_pose,
                             right_angle: float, left_angle: float,
                             right_err: bool, left_err: bool,
                             video_dims: tuple):
    """Knee angle value rendered beside each knee joint."""
    import cv2
    w, h = video_dims
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

    r_px = (int(rk.x * w) + 14, int(rk.y * h))
    l_px = (int(lk.x * w) + 14, int(lk.y * h))

    cv2.putText(frame, f"{int(right_angle)}", r_px,
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                _L_RED if right_err else _L_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{int(left_angle)}", l_px,
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                _L_RED if left_err else _L_GREEN, 2, cv2.LINE_AA)


def _lunge_hud(frame, counter: int, current_stage: str,
               score_pct: int,
               knee_over_toe_error: bool, knee_angle_error: bool,
               rep_flash_frames: int):
    """
    Full HUD overlay:
      - Dark top bar with REPS / STAGE / SCORE pills
      - Error banners at bottom
      - Orange flash + "REP!" text for ~12 frames after each new rep
    """
    import cv2
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), _L_BLACK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Exercise watermark (top-left, muted)
    cv2.putText(frame, "LUNGE", (14, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, _L_ORANGE, 1, cv2.LINE_AA)

    # REPS pill
    _lunge_pill(frame, f"REPS  {counter}", (14, 56), _L_GREEN)

    # STAGE pill
    stage_text, stage_color = _STAGE_DISPLAY.get(current_stage, ("---", _L_WHITE))
    _lunge_pill(frame, f"STAGE  {stage_text}", (115, 56), stage_color)

    # SCORE pill
    score_color = _L_GREEN if score_pct >= 80 else _L_YELLOW if score_pct >= 50 else _L_RED
    _lunge_pill(frame, f"SCORE  {score_pct}%", (330, 56), score_color)

    # Error banners (bottom)
    banner_idx = 0
    if knee_over_toe_error:
        _lunge_error_banner(frame,
                            "KNEE OVER TOE — STEP FURTHER FORWARD",
                            banner_idx, w, h)
        banner_idx += 1
    if knee_angle_error:
        _lunge_error_banner(frame,
                            "KNEE ANGLE OUT OF RANGE  [60 - 125 deg]",
                            banner_idx, w, h)

    # Rep-completion flash — fades over 12 frames
    if rep_flash_frames > 0:
        alpha = (rep_flash_frames / 12.0) * 0.38
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), _L_WHITE, -1)
        cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0, frame)

        # "REP!" text centred on screen
        font = cv2.FONT_HERSHEY_DUPLEX
        label = "REP!"
        (tw, th), _ = cv2.getTextSize(label, font, 2.6, 3)
        cv2.putText(frame, label,
                    ((w - tw) // 2, (h + th) // 2),
                    font, 2.6, _L_ORANGE, 3, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# LungeDetector
# ─────────────────────────────────────────────────────────────────────────────

class LungeDetector(ExerciseDetector):
    """
    Lunge exercise detector with annotated video output.

    Stage detection  : _LungeGCN (n_classes=3)  → I / M / D
    Error detection  : _LungeGCN (n_classes=2)  → C / L  (knee-over-toe)
    Knee angle check : geometric [60°, 125°], only at DOWN, skipped if KOT active
    Rep counting     : I → M → D = +1 rep
    Annotations      : skeleton (green/red), HUD pills, knee angles,
                       error banners, rep flash
    """

    PREDICTION_PROB_THRESHOLD = 0.80
    KNEE_ANGLE_THRESHOLD      = [60, 125]

    _STAGE_LABEL = {0: "D", 1: "I", 2: "M"}
    _ERR_LABEL   = {0: "C", 1: "L"}

    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import time
        import pickle
        import uuid
        from pathlib import Path as _Path

        # ── Load models ───────────────────────────────────────────────────
        stage_weights = parent_dir / "lunge_stage_gcn.pth"
        stage_scaler  = parent_dir / "lunge_stage_gcn_scaler.pkl"
        err_weights   = parent_dir / "lunge_err_gcn.pth"
        err_scaler    = parent_dir / "lunge_err_gcn_scaler.pkl"

        for p in (stage_weights, stage_scaler, err_weights, err_scaler):
            if not p.exists():
                raise FileNotFoundError(f"Lunge model file not found: {p}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        stage_model = _LungeGCN(n_classes=3).to(device)
        stage_model.load_state_dict(torch.load(stage_weights, map_location=device))
        stage_model.eval()

        err_model = _LungeGCN(n_classes=2).to(device)
        err_model.load_state_dict(torch.load(err_weights, map_location=device))
        err_model.eval()

        with open(stage_scaler, "rb") as f: sc_stage = pickle.load(f)
        with open(err_scaler,   "rb") as f: sc_err   = pickle.load(f)

        mp_pose = mp.solutions.pose

        # ── Open input video ──────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Output video ──────────────────────────────────────────────────
        in_path = _Path(video_path)
        if output_path:
            processed_path = output_path
        else:
            processed_path = str(
                in_path.parent / f"lunge_annotated_{uuid.uuid4().hex[:8]}.mp4"
            )

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

        # ── Session state ─────────────────────────────────────────────────
        current_stage    = ""
        counter          = 0
        score_total      = 0
        prev_counter     = 0
        rep_flash_frames = 0

        # ── THE FIX: per-rep error flags (mirrors original lunge.py) ─────
        # Original uses errors_from_this_rep (a set of error types seen THIS rep)
        # to prevent the same error type from counting more than once per rep.
        # We track them as boolean flags, reset at each rep completion — exactly
        # matching "if 'knee over toe' not in errors_from_this_rep" logic.
        rep_kot_error_seen   = False   # knee-over-toe seen this rep
        rep_angle_error_seen = False   # knee-angle error seen this rep

        # Total error counts — incremented at most ONCE per rep per type
        knee_over_toe_errors = 0
        knee_angle_errors    = 0

        start_time = time.time()

        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.8, min_tracking_confidence=0.8,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if rep_flash_frames > 0:
                    rep_flash_frames -= 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if not res.pose_landmarks:
                    writer.write(frame)
                    continue

                landmarks = res.pose_landmarks.landmark

                # ── Build landmark dict ───────────────────────────────────
                raw_row = {}
                for lm_name, lm_key in zip(_LUNGE_LM_NAMES, _LUNGE_LANDMARKS):
                    lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
                    raw_row[f"{lm_key}_x"] = lm_obj.x
                    raw_row[f"{lm_key}_y"] = lm_obj.y
                    raw_row[f"{lm_key}_z"] = lm_obj.z
                    raw_row[f"{lm_key}_v"] = lm_obj.visibility

                # ── Stage GCN ─────────────────────────────────────────────
                stage_idx, _, stage_conf = _predict_lunge_gcn(
                    raw_row, stage_model, sc_stage, device
                )
                stage_label = self._STAGE_LABEL[stage_idx]

                if stage_conf >= self.PREDICTION_PROB_THRESHOLD:
                    if stage_label == "I":
                        current_stage = "init"
                    elif stage_label == "M":
                        current_stage = "mid"
                    elif stage_label == "D":
                        if current_stage in ("init", "mid"):
                            # ── Rep completed ─────────────────────────────
                            counter          += 1
                            rep_flash_frames  = 12

                            # Score this rep (original had no scoring, we add it)
                            if rep_kot_error_seen:
                                pass              # KOT error → no points (unsafe rep)
                            elif rep_angle_error_seen:
                                score_total += 5  # angle error → half points
                            else:
                                score_total += 10 # clean rep → full points

                            # Reset per-rep flags for next rep
                            rep_kot_error_seen   = False
                            rep_angle_error_seen = False

                        current_stage = "down"

                # ── Error detection (DOWN stage only) ─────────────────────
                knee_over_toe_error = False
                knee_angle_error    = False
                right_angle = left_angle = 0.0
                right_angle_error = left_angle_error = False

                if current_stage == "down":
                    # Knee-over-toe: GCN error model
                    err_idx, _, err_conf = _predict_lunge_gcn(
                        raw_row, err_model, sc_err, device
                    )
                    err_label = self._ERR_LABEL[err_idx]

                    if err_label == "L" and err_conf >= self.PREDICTION_PROB_THRESHOLD:
                        knee_over_toe_error = True

                    # ── Count KOT error ONCE per rep (original pattern) ───
                    if knee_over_toe_error and not rep_kot_error_seen:
                        knee_over_toe_errors += 1
                        rep_kot_error_seen    = True

                    # Knee angle: geometric check, skipped if KOT active
                    if not knee_over_toe_error:
                        rh = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        rk = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                        lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                        right_angle = _calculate_angle(rh, rk, ra)
                        left_angle  = _calculate_angle(lh, lk, la)

                        lo, hi = self.KNEE_ANGLE_THRESHOLD
                        right_angle_error = not (lo <= right_angle <= hi)
                        left_angle_error  = not (lo <= left_angle  <= hi)
                        knee_angle_error  = right_angle_error or left_angle_error

                        # ── Count angle error ONCE per rep (original pattern) ──
                        if knee_angle_error and not rep_angle_error_seen:
                            knee_angle_errors    += 1
                            rep_angle_error_seen  = True

                has_error = knee_over_toe_error or knee_angle_error

                # Running score for HUD
                max_so_far = counter * 10 if counter > 0 else 10
                score_pct  = int((score_total / max_so_far) * 100)

                # ── Draw annotations (UNCHANGED from original) ────────────
                _lunge_draw_skeleton(frame, res, has_error)

                if current_stage == "down" and not knee_over_toe_error:
                    _lunge_draw_knee_angles(
                        frame, landmarks, mp_pose,
                        right_angle, left_angle,
                        right_angle_error, left_angle_error,
                        (width, height)
                    )

                _lunge_hud(
                    frame, counter, current_stage, score_pct,
                    knee_over_toe_error, knee_angle_error,
                    rep_flash_frames
                )

                writer.write(frame)

        cap.release()
        writer.release()
        duration = time.time() - start_time

        max_score    = counter * 10 if counter > 0 else 10
        score_pct    = int((score_total / max_score) * 100) if max_score > 0 else 0
        total_errors = knee_over_toe_errors + knee_angle_errors

        return {
            "score"           : score_pct,
            "duration_seconds": duration,
            "error_count"     : total_errors,
            "processed_video" : _Path(processed_path).name,
            "details": {
                "total_reps"          : counter,
                "knee_over_toe_errors": knee_over_toe_errors,
                "knee_angle_errors"   : knee_angle_errors,
            },
        }

# ─────────────────────────────────────────────────────────────────────────────
# SquatDetector
#
# Stage detection : GCN  (squat_stage_gcn.pth + squat_stage_gcn_scaler.pkl)
#                   3 classes — "down" | "middle" | "up"
# Error detection : Geometric — foot/knee placement ratios (original logic)
#                   Errors counted on TRANSITION only (not per-frame)
#                   This matches the original squat.py previous_stage pattern.
# Annotated video : Written to output_path when provided (HUD overlay)
# ─────────────────────────────────────────────────────────────────────────────

# ── Squat GCN constants ───────────────────────────────────────────────────────

_SQUAT_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]
_SQUAT_N_NODES = len(_SQUAT_LANDMARKS)   # 9
_SQUAT_N_FEATS = 4

_SQUAT_SKELETON_EDGES = [
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

_squat_node_to_idx = {n: i for i, n in enumerate(_SQUAT_LANDMARKS)}
_squat_src, _squat_dst = [], []
for _u, _v in _SQUAT_SKELETON_EDGES:
    _i, _j = _squat_node_to_idx[_u], _squat_node_to_idx[_v]
    _squat_src += [_i, _j]
    _squat_dst += [_j, _i]
_SQUAT_EDGE_INDEX = torch.tensor([_squat_src, _squat_dst], dtype=torch.long)

_SQUAT_FEATURE_COLS = [
    f"{lm}_{c}" for lm in _SQUAT_LANDMARKS for c in ["x", "y", "z", "v"]
]


# ── Squat GCN model ───────────────────────────────────────────────────────────

class _SquatGCN(nn.Module):
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
            nn.Linear(_SQUAT_N_NODES * out_feats, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),                          nn.ReLU(), nn.Dropout(dropout),
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
        x = x.view(batch.max().item() + 1, _SQUAT_N_NODES * self.out_feats)
        return self.head(x)


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize_squat_pose_row(row: dict) -> dict:
    cx = (row["left_shoulder_x"] + row["right_shoulder_x"] +
          row["left_hip_x"]      + row["right_hip_x"]) / 4
    cy = (row["left_shoulder_y"] + row["right_shoulder_y"] +
          row["left_hip_y"]      + row["right_hip_y"]) / 4
    sh_mx = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
    sh_my = (row["left_shoulder_y"] + row["right_shoulder_y"]) / 2
    hi_mx = (row["left_hip_x"]      + row["right_hip_x"])      / 2
    hi_my = (row["left_hip_y"]      + row["right_hip_y"])      / 2
    torso = ((sh_mx - hi_mx) ** 2 + (sh_my - hi_my) ** 2) ** 0.5 + 1e-6
    out = dict(row)
    for lm in _SQUAT_LANDMARKS:
        out[f"{lm}_x"] = (row[f"{lm}_x"] - cx) / torso
        out[f"{lm}_y"] = (row[f"{lm}_y"] - cy) / torso
        out[f"{lm}_z"] =  row[f"{lm}_z"]        / torso
    return out


# ── GCN stage predictor ───────────────────────────────────────────────────────

def _predict_squat_stage_gcn(raw_row, gcn_model, scaler, label_encoder,
                              device, threshold=0.6):
    norm        = _normalize_squat_pose_row(raw_row)
    feat_vec    = [[norm[col] for col in _SQUAT_FEATURE_COLS]]
    feat_scaled = scaler.transform(feat_vec)
    x_node      = torch.tensor(feat_scaled[0].reshape(_SQUAT_N_NODES, _SQUAT_N_FEATS), dtype=torch.float)
    data        = Data(x=x_node, edge_index=_SQUAT_EDGE_INDEX)
    data.batch  = torch.zeros(_SQUAT_N_NODES, dtype=torch.long)
    data        = data.to(device)
    with torch.no_grad():
        logits = gcn_model(data)
        probs  = F.softmax(logits, dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    if confidence < threshold:
        return "unknown", confidence
    return label_encoder.inverse_transform([pred_idx])[0], confidence


# ── Geometric error analysis — exact original logic ───────────────────────────

def _analyze_squat_foot_knee_placement(landmarks, stage,
                                        foot_shoulder_ratio_thresholds,
                                        knee_foot_ratio_thresholds,
                                        visibility_threshold):
    """
    Returns {"foot_placement": code, "knee_placement": code}
    Codes: -1=unknown, 0=correct, 1=too tight, 2=too wide
    Exact port of original squat.py analyze_foot_knee_placement().
    """
    import math
    import mediapipe as mp
    mp_pose = mp.solutions.pose

    result = {"foot_placement": -1, "knee_placement": -1}

    def vis(lm):    return landmarks[mp_pose.PoseLandmark[lm].value].visibility
    def xy(lm):
        p = landmarks[mp_pose.PoseLandmark[lm].value]
        return [p.x, p.y]
    def dist(a, b): return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

    if any(vis(lm) < visibility_threshold for lm in [
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_KNEE", "RIGHT_KNEE",
    ]):
        return result

    shoulder_width      = dist(xy("LEFT_SHOULDER"), xy("RIGHT_SHOULDER"))
    foot_width          = dist(xy("LEFT_FOOT_INDEX"), xy("RIGHT_FOOT_INDEX"))
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)
    lo, hi = foot_shoulder_ratio_thresholds
    if lo <= foot_shoulder_ratio <= hi:
        result["foot_placement"] = 0
    elif foot_shoulder_ratio < lo:
        result["foot_placement"] = 1
    else:
        result["foot_placement"] = 2

    # Knee only checked when feet are correct (same as original)
    if result["foot_placement"] == 0:
        knee_width      = dist(xy("LEFT_KNEE"), xy("RIGHT_KNEE"))
        knee_foot_ratio = round(knee_width / foot_width, 1)
        thresholds      = knee_foot_ratio_thresholds.get(stage)
        if thresholds is not None:
            lo_k, hi_k = thresholds
            if lo_k <= knee_foot_ratio <= hi_k:
                result["knee_placement"] = 0
            elif knee_foot_ratio < lo_k:
                result["knee_placement"] = 1
            else:
                result["knee_placement"] = 2

    return result


# ── HUD drawing helpers ───────────────────────────────────────────────────────

_SQ_GREEN  = (0,   220, 100)
_SQ_RED    = (30,   40, 220)
_SQ_YELLOW = (0,   210, 255)
_SQ_WHITE  = (255, 255, 255)
_SQ_BLACK  = (0,     0,   0)
_SQ_ORANGE = (30,  140, 255)


def _sq_pill(frame, text, pos, bg_color):
    import cv2
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), bg_color, -1)
    cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), _SQ_BLACK, 1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _SQ_BLACK, 1, cv2.LINE_AA)


def _sq_error_banner(frame, text, idx, w, h):
    import cv2
    y = h - 44 - idx * 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y-30), (w, y+10), (20, 20, 180), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"!  {text}", (14, y), cv2.FONT_HERSHEY_DUPLEX, 0.52, _SQ_WHITE, 1, cv2.LINE_AA)


def _sq_draw_hud(frame, stage, counter, score_pct, foot_label, knee_label, has_error):
    import cv2
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Title
    cv2.putText(frame, "SQUAT", (14, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, _SQ_ORANGE, 1, cv2.LINE_AA)

    # Stage pill
    stage_col = (_SQ_GREEN if stage == "up"
                 else _SQ_YELLOW if stage == "middle"
                 else _SQ_RED if stage == "down"
                 else (140, 140, 140))
    _sq_pill(frame, f"STAGE  {stage.upper()}", (14, 52), stage_col)

    # Reps pill
    _sq_pill(frame, f"REPS  {counter}", (155, 52), _SQ_GREEN)

    # Score pill
    score_col = _SQ_GREEN if score_pct >= 80 else _SQ_YELLOW if score_pct >= 50 else _SQ_RED
    _sq_pill(frame, f"SCORE  {score_pct}%", (255, 52), score_col)

    # Error banners at bottom
    banners = []
    if foot_label in ("too tight", "too wide"):
        banners.append(f"FOOT {foot_label.upper()}")
    if knee_label in ("too tight", "too wide"):
        banners.append(f"KNEE {knee_label.upper()}")
    for i, b in enumerate(banners):
        _sq_error_banner(frame, b, i, w, h)

    # Red border when error is active
    if has_error:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), _SQ_RED, 3)


def _sq_draw_skeleton(frame, pose_landmarks, mp_drawing, mp_pose, has_error):
    lm_col   = _SQ_RED    if has_error else _SQ_GREEN
    conn_col = (60, 60, 200) if has_error else (180, 255, 180)
    mp_drawing.draw_landmarks(
        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=lm_col,  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=conn_col, thickness=2),
    )


# ── SquatDetector ─────────────────────────────────────────────────────────────

class SquatDetector(ExerciseDetector):
    """
    Squat detector — RepTrack backend.

    THE FIX FOR 190 ERRORS:
    The previous version incremented total_foot_errors / total_knee_errors every
    frame an error was detected.  At 30 fps this explodes to hundreds of counts.

    The correct pattern (from original squat.py) uses a previous_stage dict:
    an error is counted only when the label TRANSITIONS into an error state.
    Consecutive frames with the same error do not increment the counter.
    This is implemented via previous_foot / previous_knee string tracking below.
    """

    PREDICTION_PROB_THRESHOLD      = 0.6
    VISIBILITY_THRESHOLD           = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS     = {
        "up":     [0.5, 1.0],
        "middle": [0.7, 1.0],
        "down":   [0.7, 1.1],
    }

    def process_video(self, video_path: str,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
        import cv2
        import mediapipe as mp
        import time
        import pickle

        mp_pose    = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # ── Load GCN ─────────────────────────────────────────────────────────
        gcn_weights = parent_dir / "squat_stage_gcn.pth"
        gcn_scaler  = parent_dir / "squat_stage_gcn_scaler.pkl"
        gcn_le      = parent_dir / "squat_stage_gcn_label_encoder.pkl"
        if not gcn_weights.exists() or not gcn_scaler.exists() or not gcn_le.exists():
            raise FileNotFoundError(
                f"Squat GCN files not found in {parent_dir}. "
                "Expected: squat_stage_gcn.pth, squat_stage_gcn_scaler.pkl, "
                "squat_stage_gcn_label_encoder.pkl"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(gcn_scaler, "rb") as f: scaler        = pickle.load(f)
        with open(gcn_le,     "rb") as f: label_encoder = pickle.load(f)
        gcn_model = _SquatGCN(n_classes=len(label_encoder.classes_)).to(device)
        gcn_model.load_state_dict(torch.load(gcn_weights, map_location=device))
        gcn_model.eval()

        lm_names = [
            "NOSE",
            "LEFT_SHOULDER",  "RIGHT_SHOULDER",
            "LEFT_HIP",       "RIGHT_HIP",
            "LEFT_KNEE",      "RIGHT_KNEE",
            "LEFT_ANKLE",     "RIGHT_ANKLE",
        ]

        # ── State ─────────────────────────────────────────────────────────────
        current_stage = "unknown"
        counter       = 0
        score_points  = 0
        rep_foot_err  = False
        rep_knee_err  = False

        # THE FIX: track previous frame's error label, count only on transition
        previous_foot = "correct"
        previous_knee = "correct"
        total_foot_errors = 0
        total_knee_errors = 0

        # ── Video I/O ─────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
                    if writer:
                        writer.write(frame)
                    continue

                landmarks = res.pose_landmarks.landmark

                # ── Feature row ───────────────────────────────────────────────
                raw_row = {}
                for lm_name in lm_names:
                    lm_obj = landmarks[mp_pose.PoseLandmark[lm_name].value]
                    col    = lm_name.lower()
                    raw_row[f"{col}_x"] = lm_obj.x
                    raw_row[f"{col}_y"] = lm_obj.y
                    raw_row[f"{col}_z"] = lm_obj.z
                    raw_row[f"{col}_v"] = lm_obj.visibility

                # ── Stage prediction ──────────────────────────────────────────
                predicted_stage, _ = _predict_squat_stage_gcn(
                    raw_row, gcn_model, scaler, label_encoder,
                    device, self.PREDICTION_PROB_THRESHOLD
                )

                # ── Rep counter: down → up ────────────────────────────────────
                if predicted_stage == "down":
                    current_stage = "down"
                elif current_stage == "down" and predicted_stage == "up":
                    current_stage = "up"
                    counter      += 1
                    if rep_foot_err and rep_knee_err:
                        score_points += 0
                    elif rep_foot_err or rep_knee_err:
                        score_points += 5
                    else:
                        score_points += 10
                    rep_foot_err = False
                    rep_knee_err = False
                elif predicted_stage not in ("unknown", ""):
                    current_stage = predicted_stage

                # ── Geometric errors ──────────────────────────────────────────
                geo = _analyze_squat_foot_knee_placement(
                    landmarks, current_stage,
                    self.FOOT_SHOULDER_RATIO_THRESHOLDS,
                    self.KNEE_FOOT_RATIO_THRESHOLDS,
                    self.VISIBILITY_THRESHOLD,
                )

                fp = geo["foot_placement"]
                kp = geo["knee_placement"]

                foot_label = {-1:"unknown", 0:"correct", 1:"too tight", 2:"too wide"}.get(fp, "unknown")
                knee_label = {-1:"unknown", 0:"correct", 1:"too tight", 2:"too wide"}.get(kp, "unknown")

                # ── TRANSITION counting (the actual fix) ──────────────────────
                if foot_label in ("too tight", "too wide"):
                    if previous_foot != foot_label:      # only on transition
                        total_foot_errors += 1
                        rep_foot_err       = True
                    previous_foot = foot_label
                else:
                    previous_foot = foot_label           # "correct" or "unknown"

                if knee_label in ("too tight", "too wide"):
                    if previous_knee != knee_label:
                        total_knee_errors += 1
                        rep_knee_err       = True
                    previous_knee = knee_label
                else:
                    previous_knee = knee_label

                has_error = foot_label in ("too tight", "too wide") or \
                            knee_label in ("too tight", "too wide")

                # ── Write annotated frame ─────────────────────────────────────
                if writer:
                    max_s    = counter * 10 if counter > 0 else 10
                    live_pct = int((score_points / max_s) * 100) if max_s > 0 else 0
                    _sq_draw_skeleton(frame, res.pose_landmarks,
                                      mp_drawing, mp_pose, has_error)
                    _sq_draw_hud(frame, current_stage, counter, live_pct,
                                 foot_label, knee_label, has_error)
                    writer.write(frame)

        cap.release()
        if writer:
            writer.release()

        duration  = time.time() - start_time
        max_score = counter * 10 if counter > 0 else 10
        score_pct = int((score_points / max_score) * 100) if max_score > 0 else 0

        return {
            "score"           : score_pct,
            "duration_seconds": duration,
            "error_count"     : total_foot_errors + total_knee_errors,
            "details": {
                "total_reps"  : counter,
                "foot_errors" : total_foot_errors,
                "knee_errors" : total_knee_errors,
            },
            "processed_video" : output_path,
        }
# ─────────────────────────────────────────────────────────────────────────────
# Registry — add lunge here
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_DETECTORS: Dict[str, ExerciseDetector] = {
    "bicep_curl": BicepCurlDetector(),
    "plank"     : PlankDetector(),
    "lunge"     : LungeDetector(),
    "squat"     : SquatDetector(),
}


def get_detector(exercise_name: str) -> ExerciseDetector:
    if exercise_name not in EXERCISE_DETECTORS:
        raise ValueError(f"No detector found for exercise: {exercise_name}")
    return EXERCISE_DETECTORS[exercise_name]


