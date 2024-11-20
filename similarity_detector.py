import numpy as np
import librosa
import logging
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass
class Match:
    start_time: float
    end_time: float
    confidence: float
    context_score: float

class SimilarityDetector:
    def __init__(self, 
                 min_duration: float = 1.0,  # Minimum match duration in seconds
                 overlap_ratio: float = 0.5,  # Window overlap ratio
                 context_size: int = 3):      # Number of windows to consider for context
        self.min_duration = min_duration
        self.overlap_ratio = overlap_ratio
        self.context_size = context_size

    def sliding_window_similarity(self, 
                                sample_chroma: np.ndarray,
                                song_chroma: np.ndarray,
                                sr: int,
                                hop_length: int = 512) -> List[Match]:
        """
        Compute similarity using sliding window approach with context awareness
        """
        logger.info("Starting sliding window similarity analysis")
        
        # Calculate window parameters
        window_length = sample_chroma.shape[1]  # Use sample length as window
        hop_size = int(window_length * (1 - self.overlap_ratio))
        
        # Initialize arrays for storing similarity scores
        num_windows = (song_chroma.shape[1] - window_length) // hop_size + 1
        similarity_scores = np.zeros(num_windows)
        
        # Compute similarity for each window
        for i in range(num_windows):
            start_idx = i * hop_size
            end_idx = start_idx + window_length
            
            # Extract window
            window = song_chroma[:, start_idx:end_idx]
            
            # Compute similarity for this window
            similarity = self._compute_window_similarity(sample_chroma, window)
            similarity_scores[i] = similarity
        
        # Find potential matches considering context
        matches = self._find_matches_with_context(
            similarity_scores,
            window_length,
            hop_size,
            sr,
            hop_length
        )
        
        logger.info(f"Found {len(matches)} potential matches")
        return matches

    def _compute_window_similarity(self, 
                                 sample_chroma: np.ndarray, 
                                 window_chroma: np.ndarray) -> float:
        """
        Compute similarity between sample and window using multiple metrics
        """
        # Normalize chromagrams
        norm_sample = librosa.util.normalize(sample_chroma, axis=0)
        norm_window = librosa.util.normalize(window_chroma, axis=0)
        
        # Compute cosine similarity
        cosine_sim = np.mean(np.sum(norm_sample * norm_window, axis=0))
        
        # Compute correlation
        correlation = np.corrcoef(norm_sample.flatten(), norm_window.flatten())[0, 1]
        
        # Combine metrics (you can adjust weights)
        similarity = 0.6 * cosine_sim + 0.4 * max(0, correlation)
        
        return float(similarity)

    def _find_matches_with_context(self,
                                 similarity_scores: np.ndarray,
                                 window_length: int,
                                 hop_size: int,
                                 sr: int,
                                 hop_length: int,
                                 base_threshold: float = 0.8) -> List[Match]:
        """
        Find matches considering surrounding context and minimum duration
        """
        matches = []
        min_frames = int(self.min_duration * sr / hop_length)
        
        # Compute context scores
        context_scores = self._compute_context_scores(similarity_scores)
        
        # Find continuous segments above threshold
        is_match = similarity_scores > base_threshold
        match_starts = np.where(np.diff(is_match.astype(int)) == 1)[0] + 1
        match_ends = np.where(np.diff(is_match.astype(int)) == -1)[0] + 1
        
        # Handle edge cases
        if len(match_starts) == 0:
            return matches
        if len(match_ends) == 0 or match_ends[-1] < match_starts[-1]:
            match_ends = np.append(match_ends, len(is_match))
        
        # Process each potential match
        for start, end in zip(match_starts, match_ends):
            # Convert frame indices to time
            duration_frames = (end - start) * hop_size
            if duration_frames < min_frames:
                continue
                
            start_time = start * hop_size * hop_length / sr
            end_time = end * hop_size * hop_length / sr
            
            # Calculate confidence and context scores
            confidence = float(np.mean(similarity_scores[start:end]))
            context_score = float(np.mean(context_scores[start:end]))
            
            matches.append(Match(
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                context_score=context_score
            ))
        
        return matches

    def _compute_context_scores(self, similarity_scores: np.ndarray) -> np.ndarray:
        """
        Compute context scores by considering surrounding windows
        """
        context_scores = np.zeros_like(similarity_scores)
        
        for i in range(len(similarity_scores)):
            # Get surrounding context
            start_idx = max(0, i - self.context_size)
            end_idx = min(len(similarity_scores), i + self.context_size + 1)
            
            # Weight context (closer windows matter more)
            weights = np.exp(-np.abs(np.arange(start_idx - i, end_idx - i)) / 2)
            context = similarity_scores[start_idx:end_idx]
            
            # Compute weighted context score
            context_scores[i] = np.average(context, weights=weights)
        
        return context_scores
