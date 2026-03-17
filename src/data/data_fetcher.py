"""Data acquisition module for FaceForensics++ and Celeb-DF datasets."""

import os
import zipfile
import shutil
from pathlib import Path
import kagglehub
import cv2
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS

class DataFetcher:
    """Handle dataset downloading and initial preprocessing."""
    
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        
    def download_faceforensics(self):
        """Download FaceForensics++ dataset using kagglehub."""
        print("Downloading FaceForensics++ dataset...")
        
        # Download the dataset
        dataset_path = kagglehub.dataset_download(
            "sourcecoke/faceforensics-c23"
        )
        
        # Move to our raw data directory
        target_path = self.raw_dir / "FaceForensics++_C23"
        if not target_path.exists():
            shutil.move(dataset_path, target_path)
            print(f"FaceForensics++ moved to: {target_path}")
        else:
            print("FaceForensics++ already exists")
            
        return target_path
    
    def download_celeba(self):
        """Download Celeb-DF dataset."""
        print("Downloading Celeb-DF dataset...")
        
        # Note: Celeb-DF might need manual download or different source
        # This is a placeholder - adjust based on actual availability
        try:
            dataset_path = kagglehub.dataset_download(
                "datasets/celeba-df"  # Verify actual dataset name
            )
            
            target_path = self.raw_dir / "CelebDF"
            if not target_path.exists():
                shutil.move(dataset_path, target_path)
                print(f"Celeb-DF moved to: {target_path}")
            else:
                print("Celeb-DF already exists")
                
            return target_path
        except Exception as e:
            print(f"Celeb-DF download failed: {e}")
            print("Please download manually from official source")
            return None
    
    def extract_sample_frames(self, video_path, output_dir, num_frames=10):
        """Extract sample frames from video for quick preprocessing."""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select evenly spaced frames
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def scan_dataset(self, dataset_path, dataset_name):
        """Scan dataset and create metadata."""
        video_files = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    full_path = Path(root) / file
                    # Determine if fake or real based on folder structure
                    parent_folder = full_path.parent.name
                    
                    is_fake = 0  # default real
                    method = "unknown"
                    
                    if dataset_name == "FaceForensics++_C23":
                        if parent_folder in DATASETS["faceforensics"]["subfolders"]:
                            if parent_folder != "original":
                                is_fake = 1
                                method = parent_folder
                            else:
                                method = "original"
                    
                    elif dataset_name == "CelebDF":
                        if parent_folder == "Celeb-synthesis":
                            is_fake = 1
                            method = "Celeb-synthesis"
                        elif parent_folder in ["Celeb-real", "YouTube-real"]:
                            is_fake = 0
                            method = parent_folder
                    
                    video_files.append({
                        'video_path': str(full_path),
                        'video_id': full_path.stem,
                        'dataset': dataset_name,
                        'method': method,
                        'label': is_fake
                    })
        
        return pd.DataFrame(video_files)

def main():
    """Test the data fetcher."""
    fetcher = DataFetcher()
    
    # Download datasets
    ff_path = fetcher.download_faceforensics()
    celeba_path = fetcher.download_celeba()
    
    # Scan and create metadata
    all_videos = []
    
    if ff_path and ff_path.exists():
        ff_df = fetcher.scan_dataset(ff_path, "FaceForensics++_C23")
        all_videos.append(ff_df)
        print(f"Found {len(ff_df)} videos in FaceForensics++")
    
    if celeba_path and celeba_path.exists():
        celeba_df = fetcher.scan_dataset(celeba_path, "CelebDF")
        all_videos.append(celeba_df)
        print(f"Found {len(celeba_df)} videos in Celeb-DF")
    
    if all_videos:
        master_df = pd.concat(all_videos, ignore_index=True)
        master_df.to_csv(RAW_DATA_DIR / "video_metadata.csv", index=False)
        print(f"Total videos: {len(master_df)}")
        print(f"Real: {len(master_df[master_df['label']==0])}")
        print(f"Fake: {len(master_df[master_df['label']==1])}")

if __name__ == "__main__":
    main()