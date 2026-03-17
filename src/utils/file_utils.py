"""File handling utilities."""

import os
import shutil
from pathlib import Path
import json
import pickle
import numpy as np
from typing import Any, Dict, List

def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def safe_remove(path: Path):
    """Safely remove file or directory."""
    if path.exists():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

def get_video_files(directory: Path, extensions: List[str] = ['.mp4', '.avi', '.mov']):
    """Get all video files in directory recursively."""
    video_files = []
    for ext in extensions:
        video_files.extend(directory.rglob(f"*{ext}"))
    return video_files

def save_json(data: Any, path: Path):
    """Save data as JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path: Path) -> Any:
    """Load JSON data."""
    with open(path, 'r') as f:
        return json.load(f)

def save_pickle(data: Any, path: Path):
    """Save data as pickle."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: Path) -> Any:
    """Load pickle data."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_file_size(path: Path) -> str:
    """Get human-readable file size."""
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space."""
    if not path.exists():
        return True
    
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    return free_gb >= required_gb