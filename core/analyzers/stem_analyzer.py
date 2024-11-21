import numpy as np
import torch
import logging
from typing import Dict, List
from .base_analyzer import BaseAnalyzer
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

logger = logging.getLogger(__name__)

class StemAnalyzer(BaseAnalyzer):
    def __init__(self, device="cpu", model_name="htdemucs"):
        """
        Initialize the stem analyzer with Demucs model.
        
        Args:
            device: Device to run the model on ("cpu" or "cuda")
            model_name: Name of the Demucs model to use
        """
        super().__init__()
        self.device = device
        try:
            self.model = get_model(model_name)
            self.model.to(device)
            logger.info(f"Loaded Demucs model {model_name} on {device}")
        except Exception as e:
            logger.error(f"Error loading Demucs model: {str(e)}")
            raise
            
    def separate(self, audio_data: np.ndarray, sr: int = 22050) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Demucs.
        
        Args:
            audio_data: Audio data to separate
            sr: Sample rate
            
        Returns:
            Dictionary of separated stems
        """
        try:
            # Convert to torch tensor and reshape for Demucs
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(1, -1)
            audio_tensor = torch.from_numpy(audio_data).to(self.device).float()
            
            # Resample if needed (Demucs expects 44.1kHz)
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(sr, 44100).to(self.device)
                audio_tensor = resampler(audio_tensor)
            
            # Apply model
            with torch.no_grad():
                separated = apply_model(self.model, audio_tensor, shifts=1, split=True)[0]
            
            # Convert back to numpy and original sample rate
            stems = {}
            stem_names = ['drums', 'bass', 'vocals', 'other']
            
            for i, name in enumerate(stem_names):
                stem_audio = separated[i].cpu().numpy()
                if sr != 44100:
                    # Resample back to original rate
                    resampler = torchaudio.transforms.Resample(44100, sr).to(self.device)
                    stem_audio = resampler(torch.from_numpy(stem_audio)).cpu().numpy()
                stems[name] = stem_audio.squeeze()
                
            return stems
            
        except Exception as e:
            logger.error(f"Error in stem separation: {str(e)}")
            return {}
            
    def analyze(self, audio_data: np.ndarray, sr: int = 22050) -> Dict:
        """
        Analyze audio by separating into stems and extracting features.
        
        Args:
            audio_data: Audio data to analyze
            sr: Sample rate
            
        Returns:
            Dictionary containing stems and features
        """
        try:
            # Separate into stems
            stems = self.separate(audio_data, sr)
            
            # Extract basic features for each stem
            stem_features = {}
            for name, stem_audio in stems.items():
                features = {
                    'rms': np.sqrt(np.mean(stem_audio**2)),
                    'peak': np.max(np.abs(stem_audio)),
                    'duration': len(stem_audio) / sr,
                    'zero_crossings': np.sum(np.abs(np.diff(np.signbit(stem_audio))))
                }
                stem_features[name] = features
            
            return {
                'stems': stems,  # Raw separated audio
                'stem_features': stem_features,  # Basic features per stem
                'metadata': {
                    'sample_rate': sr,
                    'num_stems': len(stems),
                    'stem_names': list(stems.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Error in stem analysis: {str(e)}")
            return {
                'stems': {},
                'stem_features': {},
                'metadata': {'error': str(e)}
            }
            
    def get_stem_weights(self) -> Dict[str, float]:
        """
        Get the importance weights for each stem type.
        Used in confidence calculations.
        """
        return {
            'drums': 0.3,  # Rhythm and timing
            'bass': 0.3,   # Foundation and harmony
            'vocals': 0.2, # Melody and lyrics
            'other': 0.2   # Additional elements
        }
