"""Frequency stream preprocessing using DCT transformation."""

import numpy as np
import cv2
from scipy.fftpack import dct, idct
from scipy.fft import fft2, fftshift
from pathlib import Path
from typing import Optional, Dict, Tuple
from tqdm import tqdm

class FrequencyMapper:
    """Extract frequency domain features using DCT."""
    
    def __init__(self, dct_size: int = 224):
        self.dct_size = dct_size
        
    def rgb_to_ycbcr(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to YCbCr color space."""
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        return ycbcr[:, :, 0]  # Return Y-channel (luminance)
    
    def apply_dct(self, image_patch: np.ndarray, 
                  normalize: bool = True) -> np.ndarray:
        """Apply 2D DCT to image patch."""
        # Ensure image is grayscale and float
        if len(image_patch.shape) == 3:
            if image_patch.shape[2] == 3:
                gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_patch[:, :, 0]
        else:
            gray = image_patch
            
        gray = gray.astype(np.float32)
        
        # Apply 2D DCT
        dct_coeff = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        
        if normalize:
            # Log scale for better visualization
            dct_coeff = np.log(np.abs(dct_coeff) + 1)
            
            # Normalize to [0, 1]
            dct_coeff = (dct_coeff - dct_coeff.min()) / (dct_coeff.max() - dct_coeff.min() + 1e-8)
        
        return dct_coeff
    
    def extract_frequency_features(self, face_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract various frequency domain features."""
        # Get luminance channel
        y_channel = self.rgb_to_ycbcr(face_crop)
        
        # Full DCT
        full_dct = self.apply_dct(y_channel)
        
        # High-frequency emphasis (detect artifacts)
        hf_mask = np.ones_like(full_dct)
        center = full_dct.shape[0] // 2
        hf_mask[center-20:center+20, center-20:center+20] = 0.1
        high_freq = full_dct * hf_mask
        
        # Low-frequency components
        lf_mask = np.zeros_like(full_dct)
        lf_mask[center-30:center+30, center-30:center+30] = 1
        low_freq = full_dct * lf_mask
        
        # Compute power spectrum
        f_transform = fft2(y_channel)
        f_shift = fftshift(f_transform)
        power_spectrum = np.log(np.abs(f_shift) + 1)
        power_spectrum = (power_spectrum - power_spectrum.min()) / (power_spectrum.max() - power_spectrum.min() + 1e-8)
        
        return {
            'dct_full': full_dct,
            'dct_high_freq': high_freq,
            'dct_low_freq': low_freq,
            'power_spectrum': power_spectrum,
            'y_channel': y_channel / 255.0  # Normalized luminance
        }
    
    def compute_frequency_statistics(self, dct_coeff: np.ndarray) -> Dict[str, float]:
        """Compute statistical features from DCT coefficients."""
        # Flatten and remove DC component
        coeff_flat = dct_coeff.flatten()
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(coeff_flat)),
            'std': float(np.std(coeff_flat)),
            'skewness': float(np.mean((coeff_flat - np.mean(coeff_flat))**3) / (np.std(coeff_flat)**3 + 1e-8)),
            'kurtosis': float(np.mean((coeff_flat - np.mean(coeff_flat))**4) / (np.std(coeff_flat)**4 + 1e-8)),
            'energy': float(np.sum(coeff_flat**2)),
            'entropy': float(-np.sum(coeff_flat * np.log(coeff_flat + 1e-8)))
        }
        
        # High-frequency energy ratio
        h, w = dct_coeff.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequency region (away from center)
        mask = np.ones_like(dct_coeff)
        mask[center_h-20:center_h+20, center_w-20:center_w+20] = 0
        
        hf_energy = np.sum((dct_coeff * mask)**2)
        total_energy = stats['energy']
        stats['hf_ratio'] = float(hf_energy / (total_energy + 1e-8))
        
        return stats
    
    def process_face_crops(self, face_crops: Dict[str, np.ndarray], 
                          output_dir: Path, video_id: str, frame_idx: int):
        """Process face crops and save frequency features."""
        frequency_data = {}
        
        for crop_name, crop in face_crops.items():
            if crop is not None:
                freq_features = self.extract_frequency_features(crop)
                freq_stats = self.compute_frequency_statistics(freq_features['dct_full'])
                
                frequency_data[crop_name] = {
                    'features': freq_features,
                    'statistics': freq_stats
                }
        
        # Save frequency data
        if frequency_data:
            save_path = output_dir / f"{video_id}_frame_{frame_idx:06d}_freq.npy"
            np.save(save_path, frequency_data)
        
        return frequency_data