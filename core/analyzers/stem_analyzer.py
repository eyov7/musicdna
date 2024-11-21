import numpy as np
from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer, AnalysisLevel
from .spectral_analyzer import SpectralAnalyzer
import logging
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

logger = logging.getLogger(__name__)

class StemAnalyzer(BaseAnalyzer):
    """Analyzer for stem-level features using Demucs for source separation"""
    
    def __init__(self, sample_rate: int = 22050, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(sample_rate)
        self.analysis_level = AnalysisLevel.STEM
        self.device = device
        self.stems = ['drums', 'bass', 'vocals', 'other']
        
        # Initialize Demucs model
        try:
            self.model = get_model('htdemucs').to(device)
            logger.info(f"Loaded Demucs model on {device}")
        except Exception as e:
            logger.error(f"Error loading Demucs model: {str(e)}")
            self.model = None
            
        # Initialize spectral analyzer for stem analysis
        self.spectral_analyzer = SpectralAnalyzer(sample_rate=sample_rate)
        
    def separate_stems(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Demucs
        
        Args:
            audio (np.ndarray): Audio signal to separate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of separated stems
        """
        if self.model is None:
            logger.error("Demucs model not initialized")
            return {}
            
        try:
            # Convert to torch tensor and reshape for Demucs
            audio_tensor = torch.tensor(audio).reshape(1, -1).to(self.device)
            
            # Apply source separation
            stems = apply_model(self.model, audio_tensor, shifts=1, split=True)[0]
            
            # Convert back to numpy arrays
            return {
                name: stem.cpu().numpy()
                for name, stem in zip(self.stems, stems)
            }
            
        except Exception as e:
            logger.error(f"Error separating stems: {str(e)}")
            return {}
    
    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio by first separating into stems and then extracting features
        
        Args:
            audio (np.ndarray): Audio signal to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - stems: Raw separated stems
                - features: Spectral features for each stem
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio)
        if audio is None:
            return {}
            
        try:
            # Separate into stems
            separated_stems = self.separate_stems(audio)
            if not separated_stems:
                return {}
                
            # Analyze each stem
            stem_features = {}
            for stem_name, stem_audio in separated_stems.items():
                # Extract spectral features for each stem
                features = self.spectral_analyzer.analyze(stem_audio)
                
                stem_features[stem_name] = {
                    'raw_audio': stem_audio,
                    'features': features
                }
            
            # Add metadata
            metadata = {
                'duration': len(audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'device': self.device,
                'stems': self.stems
            }
            
            result = {
                'stem_features': stem_features,
                'metadata': metadata
            }
            
            logger.info(f"Analyzed {len(stem_features)} stems")
            return result
            
        except Exception as e:
            logger.error(f"Error in stem analysis: {str(e)}")
            return {}
