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

class BicepCurlDetector(ExerciseDetector):
    """Bicep curl exercise detector"""
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process bicep curl video"""
        import cv2
        import mediapipe as mp
        import numpy as np
        import pandas as pd
        import time
        import pickle
        
        # Load models from parent directory
        model_file = parent_dir / "bicep_curl_model.pkl"
        scaler_file = parent_dir / "bicep_curl_input_scaler.pkl"
        
        if not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Model files not found in {parent_dir}")
        
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        
        # Import detection logic from bicep_improved
        import bicep_improved as bicep_module
        BicepPoseAnalysis = bicep_module.BicepPoseAnalysis
        calculate_angle = bicep_module.calculate_angle
        extract_important_keypoints = bicep_module.extract_important_keypoints
        POSTURE_ERROR_THRESHOLD = bicep_module.POSTURE_ERROR_THRESHOLD
        
        mp_pose = mp.solutions.pose
        
        lm_names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST",
            "LEFT_HIP", "RIGHT_HIP"
        ]
        
        headers = []
        for lm in lm_names:
            headers += [f"{lm.lower()}_{c}" for c in ["x", "y", "z", "v"]]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = time.time()
        
        left_arm = BicepPoseAnalysis("LEFT")
        right_arm = BicepPoseAnalysis("RIGHT")
        
        stand_posture = "C"
        previous_stand_posture = "C"
        total_reps_left = 0
        total_reps_right = 0
        score_left = 0
        score_right = 0
        
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
                    row = extract_important_keypoints(res, lm_names)
                    X = scaler.transform(pd.DataFrame([row], columns=headers))
                    predicted_posture = model.predict(X)[0]
                    prediction_probs = model.predict_proba(X)[0]
                    ml_confidence = float(np.max(prediction_probs))
                    
                    if ml_confidence >= POSTURE_ERROR_THRESHOLD:
                        stand_posture = predicted_posture
                    
                    lean_back_error = (stand_posture == "L")
                    
                    l_done, l_err, _, _ = left_arm.analyze(res.pose_landmarks.landmark, lean_back_error)
                    r_done, r_err, _, _ = right_arm.analyze(res.pose_landmarks.landmark, lean_back_error)
                    
                    if l_done:
                        total_reps_left += 1
                        if lean_back_error:
                            pass  # Bad rep
                        elif l_err:
                            score_left += 5
                        else:
                            score_left += 10
                    
                    if r_done:
                        total_reps_right += 1
                        if lean_back_error:
                            pass  # Bad rep
                        elif r_err:
                            score_right += 5
                        else:
                            score_right += 10
                    
                    previous_stand_posture = stand_posture
        
        cap.release()
        duration = time.time() - start_time
        
        total_reps = total_reps_left + total_reps_right
        total_score = score_left + score_right
        max_score = total_reps * 10 if total_reps > 0 else 10
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
                "left_reps": left_arm.get_counter(),
                "right_reps": right_arm.get_counter(),
                "total_reps": total_reps,
                "left_errors": left_arm.detected_errors,
                "right_errors": right_arm.detected_errors,
            }
        }

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
        model_file = parent_dir / "plank_model.pkl"
        scaler_file = parent_dir / "plank_input_scaler.pkl"
        
        if not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Model files not found in {parent_dir}")
        
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        
        # Import detection logic from plank_improved
        import plank_improved as plank_module
        PlankDetection = plank_module.PlankDetection
        extract_important_keypoints = plank_module.extract_important_keypoints
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
        
        fps = cap.get(cv2.CAP_PROP_FPS)
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
                    X = scaler.transform(pd.DataFrame([row], columns=headers))
                    
                    predicted_class = model.predict(X)[0]
                    prediction_probs = model.predict_proba(X)[0]
                    max_prob_idx = np.argmax(prediction_probs)
                    ml_confidence = float(prediction_probs[max_prob_idx])
                    
                    plank_detector.analyze(predicted_class, ml_confidence, current_time)
        
        cap.release()
        total_time = time.time() - session_start_time
        
        error_stats = plank_detector.get_error_stats()
        total_error_sec = error_stats['total_error_time']
        if plank_detector.error_start_time:
            total_error_sec += time.time() - plank_detector.error_start_time
        
        error_percentage = (total_error_sec / total_time * 100) if total_time > 0 else 0
        quality_score = max(0, 100 - int(error_percentage))
        
        total_errors = error_stats['low_back_count'] + error_stats['high_back_count']
        
        return {
            "score": quality_score,
            "duration_seconds": total_time,
            "error_count": total_errors,
            "details": {
                "low_back_errors": error_stats['low_back_count'],
                "high_back_errors": error_stats['high_back_count'],
                "error_time_seconds": total_error_sec,
                "error_percentage": error_percentage,
            }
        }

# Registry pattern for easy extension
EXERCISE_DETECTORS: Dict[str, ExerciseDetector] = {
    'bicep_curl': BicepCurlDetector(),
    'plank': PlankDetector(),
}

def get_detector(exercise_name: str) -> ExerciseDetector:
    """Get detector for exercise name"""
    if exercise_name not in EXERCISE_DETECTORS:
        raise ValueError(f"No detector found for exercise: {exercise_name}")
    return EXERCISE_DETECTORS[exercise_name]

