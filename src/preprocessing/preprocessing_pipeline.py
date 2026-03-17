"""Main preprocessing pipeline integrating all three streams."""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from src.config.settings import (RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                                 METADATA_DIR, DATASETS)
from src.preprocessing.spatial_extractor import SpatialExtractor
from src.preprocessing.frequency_mapper import FrequencyMapper
from src.preprocessing.temporal_loader import TemporalSequenceLoader
from src.data.data_fetcher import DataFetcher

class PreprocessingPipeline:
    """Main pipeline orchestrating all preprocessing steps."""
    
    def __init__(self, use_mps: bool = True):
        self.use_mps = use_mps and torch.backends.mps.is_available()
        print(f"MPS Available: {self.use_mps}")
        
        self.spatial_extractor = SpatialExtractor()
        self.frequency_mapper = FrequencyMapper()
        self.temporal_loader = TemporalSequenceLoader()
        self.data_fetcher = DataFetcher()
        
        # Create output directories
        self.spatial_dir = PROCESSED_DATA_DIR / "spatial"
        self.frequency_dir = PROCESSED_DATA_DIR / "frequency"
        self.temporal_dir = PROCESSED_DATA_DIR / "temporal"
        
        for dir_path in [self.spatial_dir, self.frequency_dir, self.temporal_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_or_create_metadata(self) -> pd.DataFrame:
        """Load existing metadata or create new one."""
        metadata_path = METADATA_DIR / "master_dataset.csv"
        
        if metadata_path.exists():
            print("Loading existing metadata...")
            return pd.read_csv(metadata_path)
        
        print("Creating new metadata...")
        
        # Scan datasets
        all_videos = []
        
        for dataset_name, dataset_info in DATASETS.items():
            dataset_path = RAW_DATA_DIR / dataset_info["name"]
            if dataset_path.exists():
                df = self.data_fetcher.scan_dataset(dataset_path, dataset_info["name"])
                all_videos.append(df)
                print(f"Found {len(df)} videos in {dataset_info['name']}")
        
        if not all_videos:
            raise FileNotFoundError("No datasets found. Please run data_fetcher.py first.")
        
        master_df = pd.concat(all_videos, ignore_index=True)
        
        # Train/val/test split (by video to prevent data leakage)
        unique_videos = master_df['video_id'].unique()
        train_videos, test_videos = train_test_split(
            unique_videos, test_size=0.2, random_state=42
        )
        train_videos, val_videos = train_test_split(
            train_videos, test_size=0.2, random_state=42
        )
        
        # Assign splits
        master_df['split'] = 'train'
        master_df.loc[master_df['video_id'].isin(val_videos), 'split'] = 'val'
        master_df.loc[master_df['video_id'].isin(test_videos), 'split'] = 'test'
        
        # Save metadata
        master_df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
        
        return master_df
    
    def process_video(self, video_info: dict) -> dict:
        """Process a single video through all streams."""
        video_path = Path(video_info['video_path'])
        video_id = video_info['video_id']
        label = video_info['label']
        method = video_info['method']
        
        print(f"Processing video: {video_id}")
        
        # Create video-specific directories
        video_spatial_dir = self.spatial_dir / video_id
        video_frequency_dir = self.frequency_dir / video_id
        video_temporal_dir = self.temporal_dir / video_id
        
        for dir_path in [video_spatial_dir, video_frequency_dir, video_temporal_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Extract frames and process spatially
        spatial_data = self.spatial_extractor.process_video(
            video_path, 
            video_spatial_dir,
            frame_interval=5,  # Sample every 5th frame
            max_frames=200  # Limit to 200 frames per video
        )
        
        if not spatial_data:
            print(f"No faces detected in {video_id}")
            return None
        
        # Process frequency features for each frame
        frame_metadata = []
        
        for frame_data in spatial_data:
            frame_id = frame_data['frame_id']
            frame_idx = frame_data['frame_idx']
            
            # Extract face crops
            face_crops = {
                'full_face': frame_data['full_face'],
                'eyes': frame_data['eyes'],
                'mouth': frame_data['mouth']
            }
            
            # Frequency analysis
            freq_features = self.frequency_mapper.process_face_crops(
                face_crops,
                video_frequency_dir,
                video_id,
                frame_idx
            )
            
            # Add to metadata
            frame_metadata.append({
                'frame_id': frame_id,
                'video_id': video_id,
                'frame_idx': frame_idx,
                'label': label,
                'method': method,
                'has_face': 1
            })
        
        return {
            'video_id': video_id,
            'num_frames': len(spatial_data),
            'frame_metadata': frame_metadata
        }
    
    def run_pipeline(self, num_workers: int = 4):
        """Execute the full preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        
        # Load metadata
        master_df = self.load_or_create_metadata()
        
        # Process each video
        all_frame_metadata = []
        
        # Filter to process only a subset for testing
        # For full processing, remove the head() call
        videos_to_process = master_df.head(10).to_dict('records')
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_video, video_info) 
                      for video_info in videos_to_process]
            
            for future in tqdm(futures, desc="Processing videos"):
                result = future.result()
                if result and 'frame_metadata' in result:
                    all_frame_metadata.extend(result['frame_metadata'])
        
        # Save frame-level metadata
        if all_frame_metadata:
            frame_metadata_df = pd.DataFrame(all_frame_metadata)
            frame_metadata_path = METADATA_DIR / "frame_metadata.csv"
            frame_metadata_df.to_csv(frame_metadata_path, index=False)
            print(f"Frame metadata saved to {frame_metadata_path}")
            
            # Create temporal sequences
            print("Creating temporal sequences...")
            sequences = self.temporal_loader.create_sequences_for_all_videos(
                frame_metadata_df,
                self.spatial_dir
            )
            
            # Save sequence metadata
            if sequences:
                sequences_df = pd.DataFrame(sequences)
                sequences_df.to_csv(METADATA_DIR / "sequence_metadata.csv", index=False)
                print(f"Sequence metadata saved with {len(sequences_df)} sequences")
        
        print("Preprocessing pipeline completed!")
        return frame_metadata_df if all_frame_metadata else None

def main():
    """Main execution function."""
    pipeline = PreprocessingPipeline(use_mps=True)
    frame_metadata = pipeline.run_pipeline(num_workers=4)
    
    # Print statistics
    if frame_metadata is not None:
        print("\n=== Preprocessing Statistics ===")
        print(f"Total frames processed: {len(frame_metadata)}")
        print(f"Fake frames: {len(frame_metadata[frame_metadata['label']==1])}")
        print(f"Real frames: {len(frame_metadata[frame_metadata['label']==0])}")

if __name__ == "__main__":
    main()