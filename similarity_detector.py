import numpy as np
import librosa
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SimilarityDetector:
    def __init__(self, sr: int = 22050, hop_length: int = 512, base_threshold: float = 0.7):
        """Initialize the similarity detector with given parameters"""
        self.sr = sr
        self.hop_length = hop_length
        self.base_threshold = base_threshold
        logger.info(f"Initialized SimilarityDetector with sr={sr}, hop_length={hop_length}")

    def _compute_window_similarity(self, sample_chroma: np.ndarray, window_chroma: np.ndarray) -> float:
        """Compute similarity between sample and window using cosine similarity"""
        # Normalize chromagrams
        norm_sample = librosa.util.normalize(sample_chroma, axis=0)
        norm_window = librosa.util.normalize(window_chroma, axis=0)
        
        # Compute cosine similarity
        similarity = np.mean(np.sum(norm_sample * norm_window, axis=0))
        return float(similarity)

    def detect_matches(self, sample: np.ndarray, song: np.ndarray, stems: Optional[Dict] = None) -> Dict:
        """Detect sample matches in song using chroma feature analysis"""
        try:
            # Extract chroma features
            sample_chroma = librosa.feature.chroma_cqt(y=sample, sr=self.sr, hop_length=self.hop_length)
            song_chroma = librosa.feature.chroma_cqt(y=song, sr=self.sr, hop_length=self.hop_length)
            
            # Find matches using sliding window
            matches = []
            window_length = sample_chroma.shape[1]
            
            # Calculate similarity scores
            for i in range(0, song_chroma.shape[1] - window_length, self.hop_length):
                window = song_chroma[:, i:i + window_length]
                if window.shape[1] == window_length:
                    similarity = self._compute_window_similarity(sample_chroma, window)
                    if similarity > self.base_threshold:
                        matches.append({
                            'start_time': i * self.hop_length / self.sr,
                            'duration': len(sample) / self.sr,
                            'confidence': float(similarity)
                        })
            
            # Sort matches by confidence and take top matches
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            matches = matches[:5]  # Limit to top 5 matches
            
            logger.info(f"Found {len(matches)} matches")
            return {'matches': matches}
            
        except Exception as e:
            logger.error(f"Error in match detection: {str(e)}")
            raise
