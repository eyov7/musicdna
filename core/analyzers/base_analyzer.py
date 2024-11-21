from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AnalysisLevel:
    """Enumeration of analysis levels"""
    SPECTRAL = "spectral"
    STEM = "stem"
    MIDI = "midi"
    HARMONIC = "harmonic"
    RHYTHMIC = "rhythmic"

class BaseAnalyzer(ABC):
    """Base class for all audio analyzers"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.analysis_level = None
        
    @abstractmethod
    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the audio and return features
        
        Args:
            audio (np.ndarray): Audio signal to analyze
            
        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        pass
    
    def validate_audio(self, audio: np.ndarray) -> bool:
        """
        Validate the input audio signal
        
        Args:
            audio (np.ndarray): Audio signal to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(audio, np.ndarray):
            logger.error("Input audio must be a numpy array")
            return False
        
        if len(audio.shape) != 1:
            logger.error("Input audio must be mono (1D array)")
            return False
            
        if len(audio) == 0:
            logger.error("Input audio is empty")
            return False
            
        return True
    
    def preprocess_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess the audio signal
        
        Args:
            audio (np.ndarray): Audio signal to preprocess
            
        Returns:
            Optional[np.ndarray]: Preprocessed audio or None if invalid
        """
        if not self.validate_audio(audio):
            return None
            
        # Normalize audio
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))
        
        return audio
        
    def get_analysis_level(self) -> str:
        """Get the analysis level of this analyzer"""
        if self.analysis_level is None:
            raise ValueError("Analysis level not set")
        return self.analysis_level
