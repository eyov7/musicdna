import numpy as np
import librosa
from typing import Dict, Any, Optional
from .base_analyzer import BaseAnalyzer, AnalysisLevel
import logging

logger = logging.getLogger(__name__)

class SpectralAnalyzer(BaseAnalyzer):
    """Analyzer for spectral features of audio"""
    
    def __init__(self, sample_rate: int = 22050, n_mels: int = 128,
                 n_mfcc: int = 20, hop_length: int = 512):
        super().__init__(sample_rate)
        self.analysis_level = AnalysisLevel.SPECTRAL
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        
    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral features from audio
        
        Args:
            audio (np.ndarray): Audio signal to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - mel_spectrogram: Mel-scaled spectrogram
                - mfcc: Mel-frequency cepstral coefficients
                - chroma: Chromagram
                - spectral_contrast: Spectral contrast
                - spectral_bandwidth: Spectral bandwidth
                - spectral_rolloff: Spectral rolloff
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio)
        if audio is None:
            return {}
            
        try:
            features = {}
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec)
            
            # Compute MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length
            )
            features['mfcc'] = mfcc
            
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            features['chroma'] = chroma
            
            # Compute spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            features['spectral_contrast'] = contrast
            
            # Compute spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            features['spectral_bandwidth'] = bandwidth
            
            # Compute spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            features['spectral_rolloff'] = rolloff
            
            # Add metadata
            features['metadata'] = {
                'duration': len(audio) / self.sample_rate,
                'hop_length': self.hop_length,
                'sample_rate': self.sample_rate
            }
            
            logger.info(f"Extracted spectral features: {list(features.keys())}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {str(e)}")
            return {}
