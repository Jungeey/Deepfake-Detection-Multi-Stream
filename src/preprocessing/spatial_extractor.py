"""Spatial stream preprocessing with MediaPipe face mesh."""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import torch
from tqdm import tqdm

from src.config.settings import (IMAGE_SIZE, EYE_PATCH_SIZE, 
                                 MOUTH_PATCH_SIZE, FACE_MESH_CONFIDENCE,
                                 PROCESSED_DATA_DIR)

class SpatialExtractor:
    """Extract face parts using MediaPipe Face Mesh."""
    def __init__(self, static_image_mode=True, max_num_faces=1):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Define landmark indices for different face parts
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 
                         159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 
                          387, 386, 385, 384, 398]
        
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
                     318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
                     269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
                     95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
    def get_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face landmarks from image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        landmark_points = []
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmark_points.append((x, y))
            
        return np.array(landmark_points)
    
    def get_bounding_box(self, landmarks: np.ndarray, indices: List[int], 
                        image_shape: Tuple[int, int], padding: float = 0.2) -> Tuple[int, int, int, int]:
        """Get bounding box for specific landmarks."""
        points = landmarks[indices]
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding
        w, h = x_max - x_min, y_max - y_min
        x_min = max(0, int(x_min - padding * w))
        y_min = max(0, int(y_min - padding * h))
        x_max = min(image_shape[1], int(x_max + padding * w))
        y_max = min(image_shape[0], int(y_max + padding * h))
        
        return x_min, y_min, x_max, y_max
    
    def extract_face_parts(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Extract full face, eyes, and mouth regions."""
        landmarks = self.get_face_landmarks(image)
        
        if landmarks is None:
            return {
                'full_face': None,
                'eyes': None,
                'mouth': None
            }
        
        h, w = image.shape[:2]
        
        # Extract full face (using face oval)
        x1, y1, x2, y2 = self.get_bounding_box(landmarks, self.FACE_OVAL, (h, w), padding=0.3)
        full_face = image[y1:y2, x1:x2]
        full_face = cv2.resize(full_face, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Extract eyes region
        eye_indices = self.LEFT_EYE + self.RIGHT_EYE
        x1, y1, x2, y2 = self.get_bounding_box(landmarks, eye_indices, (h, w), padding=0.5)
        eyes = image[y1:y2, x1:x2]
        if eyes.size > 0:
            eyes = cv2.resize(eyes, (EYE_PATCH_SIZE, EYE_PATCH_SIZE))
        else:
            eyes = None
        
        # Extract mouth region
        x1, y1, x2, y2 = self.get_bounding_box(landmarks, self.LIPS, (h, w), padding=0.3)
        mouth = image[y1:y2, x1:x2]
        if mouth.size > 0:
            mouth = cv2.resize(mouth, (MOUTH_PATCH_SIZE, MOUTH_PATCH_SIZE))
        else:
            mouth = None
        
        return {
            'full_face': full_face,
            'eyes': eyes,
            'mouth': mouth
        }
    
    def process_video(self, video_path: Path, output_dir: Path, 
                     frame_interval: int = 5, max_frames: int = 100):
        """Process video and extract spatial features."""
        video_name = video_path.stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames
        frame_indices = list(range(0, min(total_frames, max_frames * frame_interval), frame_interval))
        
        spatial_data = []
        
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Processing {video_name}")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract face parts
            face_parts = self.extract_face_parts(frame)
            
            # Save only if face detected
            if face_parts['full_face'] is not None:
                frame_data = {
                    'frame_id': f"{video_name}_frame_{frame_idx:06d}",
                    'video_id': video_name,
                    'frame_idx': frame_idx,
                    'full_face': face_parts['full_face'],
                    'eyes': face_parts['eyes'],
                    'mouth': face_parts['mouth']
                }
                spatial_data.append(frame_data)
                
                # Save individual crops (optional, can also save as numpy)
                if len(spatial_data) % 10 == 0:
                    np.save(video_output_dir / f"frame_{frame_idx:06d}_crops.npy", 
                           {'full_face': face_parts['full_face'],
                            'eyes': face_parts['eyes'],
                            'mouth': face_parts['mouth']})
        
        cap.release()
        
        # Save all spatial data for this video
        if spatial_data:
            np.save(video_output_dir / "all_crops.npy", spatial_data)
        
        return spatial_data
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        self.face_mesh.close()