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
        """Initialize the stem analyzer with Demucs model."""
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.stem_names = ['drums', 'bass', 'vocals', 'other']
        
        try:
            self.model = get_model(model_name)
            self.model.to(device)
            logger.info(f"Loaded Demucs model {model_name} on {device}")
        except Exception as e:
            logger.error(f"Error loading Demucs model: {str(e)}")
            raise

    def separate(self, audio_data: np.ndarray, sr: int = 22050) -> Dict[str, np.ndarray]:
        """Separate audio into stems using Demucs."""
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
                separated = apply_model(self.model, audio_tensor, shifts=1, split=True)
                separated = separated[0]  # Get first batch
            
            # Convert back to numpy and original sample rate
            stems = {}
            for i, name in enumerate(self.stem_names):
                stem_audio = separated[i].cpu().numpy()
                if sr != 44100:
                    # Resample back to original rate
                    resampler = torchaudio.transforms.Resample(44100, sr).to(self.device)
                    stem_audio = resampler(torch.from_numpy(stem_audio)).cpu().numpy()
                stems[name] = stem_audio.squeeze()
            
            return stems
            
        except Exception as e:
            logger.error(f"Error in stem separation: {str(e)}")
            return {name: np.zeros(1) for name in self.stem_names}  # Return empty stems on error

    def analyze(self, audio_data: np.ndarray, sr: int = 22050) -> Dict:
        """Analyze audio by separating into stems and extracting features."""
        try:
            # Separate into stems
            stems = self.separate(audio_data, sr)
            
            # Extract basic features for each stem
            stem_features = {}
            for name, stem_audio in stems.items():
                features = {
                    'rms': float(np.sqrt(np.mean(stem_audio**2))),
                    'peak': float(np.max(np.abs(stem_audio))),
                    'duration': float(len(stem_audio) / sr),
                    'zero_crossings': int(np.sum(np.abs(np.diff(np.signbit(stem_audio)))))
                }
                stem_features[name] = {
                    'audio': stem_audio,
                    'features': features
                }
            
            return {
                'stems': stem_features,
                'metadata': {
                    'sample_rate': sr,
                    'num_stems': len(stems),
                    'stem_names': self.stem_names,
                    'model': self.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error in stem analysis: {str(e)}")
            return {
                'stems': {name: {'audio': np.zeros(1), 'features': {}} for name in self.stem_names},
                'metadata': {'error': str(e)}
            }

    def get_stem_weights(self) -> Dict[str, float]:
        """Get the importance weights for each stem type."""
        return {
            'drums': 0.3,  # Rhythm and timing
            'bass': 0.3,   # Foundation and harmony
            'vocals': 0.2, # Melody and lyrics
            'other': 0.2   # Additional elements
        }
