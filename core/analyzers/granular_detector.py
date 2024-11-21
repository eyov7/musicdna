import numpy as np
import logging
from typing import Dict, Any, List, Optional
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class GranularSampleDetector(BaseAnalyzer):
    def __init__(self, stem_weights: Optional[Dict[str, float]] = None):
        """Initialize the granular sample detector."""
        super().__init__()
        self.stem_weights = stem_weights or {
            'drums': 0.3,
            'bass': 0.3,
            'vocals': 0.2,
            'other': 0.2
        }
        self.sample_rate = 22050  # Default sample rate

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio using granular detection."""
        try:
            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Ensure 2D array
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
                
            return self._analyze_impl(audio)
            
        except Exception as e:
            logger.error(f"Error in sample analysis: {str(e)}")
            return self._create_empty_result()

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result structure."""
        return {
            'confidence': 0.0,
            'matches': [],
            'transformations': [],
            'metadata': {
                'sample_rate': self.sample_rate,
                'error': 'Analysis failed'
            }
        }

    def _analyze_impl(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Internal implementation of granular analysis."""
        features = self._extract_features(audio_data)
        
        return {
            'features': features,
            'metadata': {
                'sample_rate': self.sample_rate,
                'duration': float(len(audio_data[0]) / self.sample_rate)
            }
        }

    def find_in_track(self, sample_analysis: Dict[str, Any], track_audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Find sample occurrences in a track."""
        try:
            # Convert track_audio to numpy array if needed
            if not isinstance(track_audio, np.ndarray):
                track_audio = np.array(track_audio)
                
            # Ensure 2D array
            if len(track_audio.shape) == 1:
                track_audio = track_audio.reshape(1, -1)
                
            matches = []
            
            # Handle case where stem analysis is not available
            if 'stems' not in sample_analysis:
                logger.warning("Stem analysis not available - falling back to full audio analysis")
                confidence = self._compare_audio(
                    sample_analysis.get('features', {}),
                    self._extract_features(track_audio)
                )
                if confidence > 0.7:  # Threshold for match
                    matches.append({
                        'start_time': 0,
                        'confidence': confidence,
                        'transformations': self._detect_transformations(sample_analysis, track_audio)
                    })
                return matches

            # Process each stem if available
            for stem_name, stem_weight in self.stem_weights.items():
                if stem_name not in sample_analysis.get('stems', {}):
                    continue

                stem_confidence = self._compare_stems(
                    sample_analysis['stems'][stem_name],
                    track_audio,
                    stem_weight
                )
                
                if stem_confidence > 0.7:  # Threshold for stem match
                    transformations = self._detect_transformations(
                        sample_analysis['stems'][stem_name],
                        track_audio
                    )
                    matches.append({
                        'stem': stem_name,
                        'confidence': stem_confidence,
                        'transformations': transformations
                    })

            return matches
            
        except Exception as e:
            logger.error(f"Error in find_in_track: {str(e)}")
            return []

    def _extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract audio features for comparison."""
        try:
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
                
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
                
            return {
                'rms': float(np.sqrt(np.mean(audio**2))),
                'peak': float(np.max(np.abs(audio))),
                'duration': float(len(audio[0]) / self.sample_rate),
                'zero_crossings': int(np.sum(np.abs(np.diff(np.signbit(audio[0])))))
            }
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return {
                'rms': 0.0,
                'peak': 0.0,
                'duration': 0.0,
                'zero_crossings': 0
            }

    def _compare_audio(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Compare two sets of audio features."""
        try:
            if not features1 or not features2:
                return 0.0

            # Simple feature comparison
            rms_diff = abs(features1.get('rms', 0) - features2.get('rms', 0))
            peak_diff = abs(features1.get('peak', 0) - features2.get('peak', 0))
            
            # Normalize differences
            confidence = 1.0 - (rms_diff + peak_diff) / 2
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error in audio comparison: {str(e)}")
            return 0.0

    def _compare_stems(self, stem1: Dict[str, Any], stem2: np.ndarray, weight: float) -> float:
        """Compare two stems with weighting."""
        try:
            features1 = stem1.get('features', {})
            features2 = self._extract_features(stem2)
            
            base_confidence = self._compare_audio(features1, features2)
            return base_confidence * weight
            
        except Exception as e:
            logger.error(f"Error in stem comparison: {str(e)}")
            return 0.0

    def _detect_transformations(self, sample: Dict[str, Any], track: np.ndarray) -> List[str]:
        """Detect audio transformations between sample and track."""
        try:
            transformations = []
            
            # Basic transformation detection
            sample_features = sample.get('features', {})
            track_features = self._extract_features(track)
            
            # Detect pitch shift
            if abs(sample_features.get('zero_crossings', 0) - track_features.get('zero_crossings', 0)) > 100:
                transformations.append('pitch_shift')
                
            # Detect time stretch
            duration_ratio = track_features.get('duration', 0) / sample_features.get('duration', 1)
            if abs(1 - duration_ratio) > 0.1:  # 10% threshold
                transformations.append('time_stretch')
                
            return transformations
            
        except Exception as e:
            logger.error(f"Error in transformation detection: {str(e)}")
            return []
