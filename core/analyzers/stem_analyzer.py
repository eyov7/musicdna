import numpy as np
import torch
import logging
from typing import Dict, List, Any
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
        self.sample_rate = 22050
        
        try:
            self.model = get_model(model_name)
            self.model.to(device)
            logger.info(f"Loaded Demucs model {model_name} on {device}")
        except Exception as e:
            logger.error(f"Error loading Demucs model: {str(e)}")
            raise

    def validate_audio(self, audio: np.ndarray) -> bool:
        """Validate the input audio."""
        return isinstance(audio, np.ndarray) and len(audio.shape) <= 2

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """Main analysis method required by BaseAnalyzer."""
        if not self.validate_audio(audio):
            logger.error("Invalid audio input")
            return self._create_empty_result()
            
        try:
            return self._analyze_impl(audio, self.sample_rate)
        except Exception as e:
            logger.error(f"Error in stem analysis: {str(e)}")
            return self._create_empty_result()

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result structure."""
        return {
            'stems': {
                name: {
                    'audio': np.zeros(1),
                    'features': {
                        'rms': 0.0,
                        'peak': 0.0,
                        'duration': 0.0,
                        'zero_crossings': 0
                    }
                }
                for name in self.stem_names
            },
            'metadata': {
                'sample_rate': self.sample_rate,
                'model': self.model_name,
                'error': 'Analysis failed'
            }
        }

    def _analyze_impl(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Internal implementation of stem analysis."""
        # Ensure audio is 2D (batch, samples)
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).to(self.device).float()
        
        # Resample if needed (Demucs expects 44.1kHz)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100).to(self.device)
            audio_tensor = resampler(audio_tensor)
        
        # Apply model
        with torch.no_grad():
            separated = apply_model(self.model, audio_tensor, shifts=1, split=True)[0]
            
        # Process each stem
        stems = {}
        for i, name in enumerate(self.stem_names):
            # Get stem audio
            stem_audio = separated[i].cpu().numpy()
            
            # Resample back if needed
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(44100, sr).to(self.device)
                stem_audio = resampler(torch.from_numpy(stem_audio)).cpu().numpy()
            
            stem_audio = stem_audio.squeeze()
            
            # Calculate features
            features = {
                'rms': float(np.sqrt(np.mean(stem_audio**2))),
                'peak': float(np.max(np.abs(stem_audio))),
                'duration': float(len(stem_audio) / sr),
                'zero_crossings': int(np.sum(np.abs(np.diff(np.signbit(stem_audio)))))
            }
            
            stems[name] = {
                'audio': stem_audio,
                'features': features
            }
        
        return {
            'stems': stems,
            'metadata': {
                'sample_rate': sr,
                'num_stems': len(stems),
                'stem_names': self.stem_names,
                'model': self.model_name,
                'duration': float(len(audio_data[0]) / sr)
            }
        }

    def get_stem_weights(self) -> Dict[str, float]:
        """Get the importance weights for each stem type."""
        return {
            'drums': 0.3,  # Rhythm and timing
            'bass': 0.3,   # Foundation and harmony
            'vocals': 0.2, # Melody and lyrics
            'other': 0.2   # Additional elements
        }
