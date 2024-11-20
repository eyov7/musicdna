import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from scipy import signal

logger = logging.getLogger(__name__)

class FrequencyAnalyzer:
    def __init__(self,
                 n_bands: int = 8,
                 min_freq: float = 20.0,
                 max_freq: float = 20000.0,
                 sr: int = 44100):
        """
        Initialize frequency band analyzer
        
        Args:
            n_bands: Number of frequency bands to analyze
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
            sr: Sample rate in Hz
        """
        self.n_bands = n_bands
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sr = sr
        
        # Calculate band frequencies using mel scale
        self.band_freqs = librosa.mel_frequencies(
            n_mels=n_bands + 1,
            fmin=min_freq,
            fmax=max_freq,
            htk=True
        )
        
        # Create band filters
        self.filters = self._create_band_filters()
        
        logger.info(f"Initialized FrequencyAnalyzer with {n_bands} bands")

    def _create_band_filters(self) -> List[np.ndarray]:
        """Create bandpass filters for each frequency band"""
        filters = []
        nyquist = self.sr / 2
        
        for i in range(self.n_bands):
            low = self.band_freqs[i] / nyquist
            high = self.band_freqs[i + 1] / nyquist
            
            # Create bandpass filter
            b, a = signal.butter(4, [low, high], btype='band')
            filters.append((b, a))
        
        return filters

    def analyze_frequency_content(self, 
                                audio: np.ndarray,
                                frame_length: int = 2048,
                                hop_length: int = 512) -> np.ndarray:
        """
        Analyze frequency content per band over time
        
        Returns:
            Array of shape (n_bands, n_frames) containing frequency energy
        """
        try:
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Initialize output array
            n_frames = 1 + int((len(audio) - frame_length) / hop_length)
            band_energies = np.zeros((self.n_bands, n_frames))
            
            # Analyze each band
            for i, (b, a) in enumerate(self.filters):
                # Apply bandpass filter
                filtered = signal.filtfilt(b, a, audio)
                
                # Calculate RMS energy in frames
                band_energies[i] = librosa.feature.rms(
                    y=filtered,
                    frame_length=frame_length,
                    hop_length=hop_length
                )[0]
            
            # Normalize energies
            band_energies = librosa.util.normalize(band_energies, axis=1)
            
            logger.info("Frequency content analysis completed")
            return band_energies
            
        except Exception as e:
            logger.error(f"Error in frequency content analysis: {str(e)}")
            raise

    def calculate_band_weights(self,
                             sample_audio: np.ndarray,
                             reference_audio: Optional[np.ndarray] = None,
                             frame_length: int = 2048,
                             hop_length: int = 512) -> np.ndarray:
        """
        Calculate importance weights for each frequency band
        
        Args:
            sample_audio: Audio of the sample to detect
            reference_audio: Optional reference audio to compare against
            
        Returns:
            Array of weights for each frequency band
        """
        try:
            # Analyze sample frequency content
            sample_freqs = self.analyze_frequency_content(
                sample_audio,
                frame_length,
                hop_length
            )
            
            if reference_audio is not None:
                # Compare with reference to find distinctive bands
                ref_freqs = self.analyze_frequency_content(
                    reference_audio,
                    frame_length,
                    hop_length
                )
                
                # Calculate distinctiveness as KL divergence
                weights = np.zeros(self.n_bands)
                for i in range(self.n_bands):
                    weights[i] = self._kl_divergence(
                        sample_freqs[i],
                        ref_freqs[i]
                    )
            else:
                # Use variance as weight if no reference
                weights = np.var(sample_freqs, axis=1)
            
            # Normalize weights
            weights = librosa.util.normalize(weights)
            
            logger.info("Band weight calculation completed")
            return weights
            
        except Exception as e:
            logger.error(f"Error in band weight calculation: {str(e)}")
            raise

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions"""
        # Add small constant to avoid division by zero
        eps = 1e-10
        p = p + eps
        q = q + eps
        
        # Normalize to create proper distributions
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))

    def apply_band_weights(self,
                         features: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
        """
        Apply frequency band weights to feature matrix
        
        Args:
            features: Feature matrix of shape (n_bands, n_frames)
            weights: Weight vector of shape (n_bands,)
            
        Returns:
            Weighted feature matrix
        """
        try:
            # Reshape weights for broadcasting
            weights = weights.reshape(-1, 1)
            
            # Apply weights
            weighted_features = features * weights
            
            logger.info("Applied frequency band weights to features")
            return weighted_features
            
        except Exception as e:
            logger.error(f"Error applying band weights: {str(e)}")
            raise
