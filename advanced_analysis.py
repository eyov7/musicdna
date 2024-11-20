import numpy as np
import librosa
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class MixAnalyzer:
    def __init__(self, 
                 base_threshold: float = 0.8,
                 max_threshold_reduction: float = 0.3,
                 density_sensitivity: float = 0.5):
        self.base_threshold = base_threshold
        self.max_threshold_reduction = max_threshold_reduction
        self.density_sensitivity = density_sensitivity

    def calculate_mix_density(self, 
                            stems: Dict[str, np.ndarray],
                            frame_length: int = 2048,
                            hop_length: int = 512) -> np.ndarray:
        """
        Calculate mix density over time using stem information
        
        Returns density scores normalized between 0 and 1
        where 1 indicates maximum density (all stems active)
        """
        try:
            # Calculate RMS energy for each stem
            stem_energies = {}
            for name, audio in stems.items():
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=0)
                
                # Get RMS energy
                energy = librosa.feature.rms(
                    y=audio,
                    frame_length=frame_length,
                    hop_length=hop_length
                )[0]
                
                # Normalize energy
                energy = librosa.util.normalize(energy)
                stem_energies[name] = energy

            # Calculate number of active stems per frame
            threshold = 0.1  # Energy threshold for considering a stem "active"
            active_stems = np.zeros_like(list(stem_energies.values())[0])
            
            for energy in stem_energies.values():
                active_stems += (energy > threshold).astype(float)
            
            # Normalize by total number of stems
            density = active_stems / len(stems)
            
            logger.info("Mix density analysis completed successfully")
            return density
            
        except Exception as e:
            logger.error(f"Error in mix density calculation: {str(e)}")
            raise

    def get_dynamic_threshold(self, 
                            mix_density: np.ndarray,
                            window_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate dynamic threshold based on mix density
        
        Returns an array of thresholds for each time frame
        """
        try:
            # Apply smoothing if window_size is provided
            if window_size:
                kernel = np.ones(window_size) / window_size
                smoothed_density = np.convolve(mix_density, kernel, mode='same')
            else:
                smoothed_density = mix_density
            
            # Calculate dynamic threshold
            # Reduce threshold more in dense sections
            threshold_reduction = smoothed_density * self.max_threshold_reduction
            dynamic_threshold = self.base_threshold * (1 - threshold_reduction)
            
            logger.info("Dynamic threshold calculation completed")
            return dynamic_threshold
            
        except Exception as e:
            logger.error(f"Error in dynamic threshold calculation: {str(e)}")
            raise

    def adjust_confidence_scores(self,
                               confidence_scores: np.ndarray,
                               mix_density: np.ndarray) -> np.ndarray:
        """
        Adjust confidence scores based on mix density
        
        Boosts scores in dense sections to compensate for masking effects
        """
        try:
            # Calculate boost factor based on density
            density_factor = 1 + (mix_density * self.density_sensitivity)
            
            # Apply boost to confidence scores
            adjusted_scores = confidence_scores * density_factor
            
            # Normalize to keep scores in reasonable range
            adjusted_scores = librosa.util.normalize(adjusted_scores)
            
            logger.info("Confidence score adjustment completed")
            return adjusted_scores
            
        except Exception as e:
            logger.error(f"Error in confidence score adjustment: {str(e)}")
            raise

    def analyze_section(self,
                       stems: Dict[str, np.ndarray],
                       frame_length: int = 2048,
                       hop_length: int = 512,
                       smoothing_window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform complete mix analysis returning density and dynamic threshold
        """
        try:
            # Calculate mix density
            density = self.calculate_mix_density(
                stems,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            # Calculate dynamic threshold
            threshold = self.get_dynamic_threshold(
                density,
                window_size=smoothing_window
            )
            
            return density, threshold
            
        except Exception as e:
            logger.error(f"Error in section analysis: {str(e)}")
            raise
