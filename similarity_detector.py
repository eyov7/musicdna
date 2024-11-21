import numpy as np
import librosa
import logging
from typing import Dict, List, Optional
from demucs_handler import DemucsProcessor
from basic_pitch_handler import BasicPitchConverter

logger = logging.getLogger(__name__)

class SimilarityDetector:
    def __init__(self, sr: int = 22050, hop_length: int = 512, base_threshold: float = 0.7):
        """Initialize the similarity detector with given parameters"""
        self.sr = sr
        self.hop_length = hop_length
        self.base_threshold = base_threshold
        
        # Initialize advanced processors
        self.demucs = DemucsProcessor()
        self.basic_pitch = BasicPitchConverter()
        
        logger.info(f"Initialized SimilarityDetector with sr={sr}, hop_length={hop_length}")

    def _compute_basic_similarity(self, sample_chroma: np.ndarray, window_chroma: np.ndarray) -> float:
        """Compute similarity between sample and window using cosine similarity"""
        # Normalize chromagrams
        norm_sample = librosa.util.normalize(sample_chroma, axis=0)
        norm_window = librosa.util.normalize(window_chroma, axis=0)
        
        # Compute cosine similarity
        similarity = np.mean(np.sum(norm_sample * norm_window, axis=0))
        return float(similarity)

    def _compute_advanced_similarity(self, sample_stems: Dict[str, np.ndarray], 
                                   window_stems: Dict[str, np.ndarray]) -> float:
        """Compute similarity using stem separation and pitch tracking"""
        total_similarity = 0.0
        weights = {'vocals': 0.4, 'drums': 0.2, 'bass': 0.2, 'other': 0.2}
        
        for stem_name, weight in weights.items():
            if stem_name in sample_stems and stem_name in window_stems:
                # Get MIDI features for both
                sample_midi = self.basic_pitch.audio_to_midi(sample_stems[stem_name])
                window_midi = self.basic_pitch.audio_to_midi(window_stems[stem_name])
                
                # Compare MIDI features
                midi_similarity = np.mean(np.abs(sample_midi - window_midi))
                total_similarity += weight * midi_similarity
        
        return float(total_similarity)

    def detect_matches(self, sample: np.ndarray, song: np.ndarray, use_advanced: bool = True) -> Dict:
        """Detect sample matches using both basic and advanced techniques"""
        try:
            matches = []
            
            # Basic chroma-based detection
            sample_chroma = librosa.feature.chroma_cqt(y=sample, sr=self.sr, hop_length=self.hop_length)
            song_chroma = librosa.feature.chroma_cqt(y=song, sr=self.sr, hop_length=self.hop_length)
            window_length = sample_chroma.shape[1]
            
            # Advanced stem-based detection if requested
            if use_advanced:
                sample_stems = self.demucs.separate_stems(sample)
                song_stems = self.demucs.separate_stems(song)
            
            # Find matches using sliding window
            for i in range(0, song_chroma.shape[1] - window_length, self.hop_length):
                window = song_chroma[:, i:i + window_length]
                if window.shape[1] == window_length:
                    # Basic similarity
                    basic_similarity = self._compute_basic_similarity(sample_chroma, window)
                    
                    # Advanced similarity if enabled
                    if use_advanced:
                        window_audio = song[i:i + len(sample)]
                        window_stems = self.demucs.separate_stems(window_audio)
                        advanced_similarity = self._compute_advanced_similarity(sample_stems, window_stems)
                        # Combine similarities (weighted average)
                        similarity = 0.6 * basic_similarity + 0.4 * advanced_similarity
                    else:
                        similarity = basic_similarity
                    
                    if similarity > self.base_threshold:
                        matches.append({
                            'start_time': i * self.hop_length / self.sr,
                            'duration': len(sample) / self.sr,
                            'confidence': float(similarity),
                            'method': 'hybrid' if use_advanced else 'basic'
                        })
            
            # Sort matches by confidence and take top matches
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            matches = matches[:5]  # Limit to top 5 matches
            
            logger.info(f"Found {len(matches)} matches using {'hybrid' if use_advanced else 'basic'} method")
            return {'matches': matches}
            
        except Exception as e:
            logger.error(f"Error in match detection: {str(e)}")
            raise
