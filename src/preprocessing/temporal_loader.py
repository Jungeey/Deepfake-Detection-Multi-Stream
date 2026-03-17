"""Temporal stream preprocessing for sequence creation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

from src.config.settings import FRAMES_PER_SEQUENCE, STRIDE

class TemporalSequenceLoader:
    """Create temporal sequences from video frames."""
    
    def __init__(self, frames_per_sequence: int = FRAMES_PER_SEQUENCE, 
                 stride: int = STRIDE):
        self.frames_per_sequence = frames_per_sequence
        self.stride = stride
        
    def create_sequences(self, frame_indices: List[int]) -> List[List[int]]:
        """Create overlapping sequences from frame indices."""
        sequences = []
        
        if len(frame_indices) < self.frames_per_sequence:
            return [frame_indices]  # Return single sequence if not enough frames
        
        for start_idx in range(0, len(frame_indices) - self.frames_per_sequence + 1, self.stride):
            sequence = frame_indices[start_idx:start_idx + self.frames_per_sequence]
            sequences.append(sequence)
        
        return sequences
    
    def extract_optical_flow(self, prev_frame: np.ndarray, 
                            curr_frame: np.ndarray) -> np.ndarray:
        """Extract optical flow between two frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        # Convert flow to magnitude and angle
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Stack as 2-channel image
        flow_image = np.stack([magnitude, angle], axis=-1)
        
        return flow_image
    
    def compute_motion_features(self, frame_sequence: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute motion features from frame sequence."""
        motion_features = {
            'optical_flow': [],
            'frame_differences': [],
            'motion_magnitudes': []
        }
        
        for i in range(len(frame_sequence) - 1):
            # Optical flow
            flow = self.extract_optical_flow(frame_sequence[i], frame_sequence[i + 1])
            motion_features['optical_flow'].append(flow)
            
            # Frame difference
            diff = np.abs(frame_sequence[i + 1].astype(np.float32) - 
                         frame_sequence[i].astype(np.float32))
            diff = np.mean(diff, axis=-1) if diff.ndim == 3 else diff
            motion_features['frame_differences'].append(diff)
            
            # Motion magnitude (mean of optical flow magnitude)
            motion_features['motion_magnitudes'].append(np.mean(flow[..., 0]))
        
        # Stack features
        motion_features['optical_flow'] = np.stack(motion_features['optical_flow'])
        motion_features['frame_differences'] = np.stack(motion_features['frame_differences'])
        motion_features['motion_magnitudes'] = np.array(motion_features['motion_magnitudes'])
        
        return motion_features

class TemporalDataset(Dataset):
    """PyTorch dataset for temporal sequences."""
    
    def __init__(self, metadata_df: pd.DataFrame, frames_dir: Path,
                 transform=None, frames_per_sequence: int = FRAMES_PER_SEQUENCE):
        self.metadata = metadata_df
        self.frames_dir = frames_dir
        self.transform = transform
        self.frames_per_sequence = frames_per_sequence
        
        # Group frames by video
        self.video_groups = self.metadata.groupby('video_id')
        
        # Create sequences
        self.sequences = []
        self.labels = []
        self.video_ids = []
        
        print("Creating temporal sequences...")
        for video_id, group in tqdm(self.video_groups):
            # Sort by frame index
            group = group.sort_values('frame_idx')
            frame_indices = group['frame_idx'].tolist()
            label = group['label'].iloc[0]  # All frames same label
            
            # Create sequences
            sequences = self._create_sequences(frame_indices)
            
            for seq in sequences:
                self.sequences.append((video_id, seq))
                self.labels.append(label)
                self.video_ids.append(video_id)
    
    def _create_sequences(self, frame_indices: List[int]) -> List[List[int]]:
        """Create overlapping sequences."""
        sequences = []
        
        if len(frame_indices) < self.frames_per_sequence:
            return [frame_indices]
        
        stride = max(1, self.frames_per_sequence // 2)
        
        for start_idx in range(0, len(frame_indices) - self.frames_per_sequence + 1, stride):
            sequence = frame_indices[start_idx:start_idx + self.frames_per_sequence]
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        video_id, frame_indices = self.sequences[idx]
        
        # Load frames
        frames = []
        for frame_idx in frame_indices:
            # Construct frame path based on your saved format
            frame_path = self.frames_dir / video_id / f"frame_{frame_idx:06d}_crops.npy"
            
            if frame_path.exists():
                frame_data = np.load(frame_path, allow_pickle=True).item()
                # Use full face crop
                frame = frame_data['full_face']
            else:
                # Fallback to empty array
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # Stack frames: (seq_len, C, H, W)
        frames = torch.stack(frames) if isinstance(frames[0], torch.Tensor) else torch.tensor(np.stack(frames))
        
        return {
            'frames': frames,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'video_id': self.video_ids[idx],
            'sequence_indices': torch.tensor(frame_indices)
        }