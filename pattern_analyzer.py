import numpy as np
import librosa
import logging
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    def __init__(self,
                 min_pattern_length: float = 1.0,
                 max_pattern_gap: float = 0.5,
                 similarity_threshold: float = 0.85):
        """
        Initialize pattern analyzer for detecting repeated sections
        
        Args:
            min_pattern_length: Minimum length of a pattern in seconds
            max_pattern_gap: Maximum gap between similar sections to be considered part of same pattern
            similarity_threshold: Threshold for considering sections similar
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_gap = max_pattern_gap
        self.similarity_threshold = similarity_threshold

    def find_repetitions(self,
                        features: np.ndarray,
                        sr: int = 44100,
                        hop_length: int = 512) -> List[Dict]:
        """
        Find repeated patterns in feature sequence
        
        Args:
            features: Feature matrix (e.g., chroma or mel spectrogram)
            sr: Sample rate
            hop_length: Number of samples between frames
            
        Returns:
            List of dictionaries containing pattern information
        """
        try:
            # Calculate self-similarity matrix
            S = self._compute_similarity_matrix(features)
            
            # Find diagonal stripes (indicating repeated patterns)
            patterns = self._find_diagonal_patterns(S)
            
            # Convert frame indices to time
            patterns = self._convert_to_time(patterns, sr, hop_length)
            
            # Group similar patterns
            grouped_patterns = self._group_patterns(patterns)
            
            logger.info(f"Found {len(grouped_patterns)} distinct patterns")
            return grouped_patterns
            
        except Exception as e:
            logger.error(f"Error finding repetitions: {str(e)}")
            raise

    def _compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute self-similarity matrix"""
        # Normalize features
        features_norm = librosa.util.normalize(features, axis=0)
        
        # Compute cosine similarity
        S = np.dot(features_norm.T, features_norm)
        
        return S

    def _find_diagonal_patterns(self, S: np.ndarray) -> List[Dict]:
        """Find diagonal stripes in similarity matrix"""
        patterns = []
        n = S.shape[0]
        
        # Look for diagonals offset from main diagonal
        for diag_offset in range(1, n-int(self.min_pattern_length)):
            diagonal = np.diagonal(S, offset=diag_offset)
            
            # Find peaks in diagonal (high similarity regions)
            peaks, properties = find_peaks(
                diagonal,
                height=self.similarity_threshold,
                distance=int(self.min_pattern_length)
            )
            
            # Convert peaks to pattern segments
            for peak_idx in peaks:
                pattern = {
                    'start1': peak_idx,
                    'start2': peak_idx + diag_offset,
                    'length': self._get_pattern_length(S, peak_idx, diag_offset),
                    'similarity': float(diagonal[peak_idx])
                }
                patterns.append(pattern)
        
        return patterns

    def _get_pattern_length(self,
                          S: np.ndarray,
                          start_idx: int,
                          offset: int) -> int:
        """Find length of similar diagonal starting at given point"""
        n = S.shape[0]
        length = 0
        
        while (start_idx + length < n and
               start_idx + offset + length < n and
               S[start_idx + length, start_idx + offset + length] >= self.similarity_threshold):
            length += 1
            
        return length

    def _convert_to_time(self,
                        patterns: List[Dict],
                        sr: int,
                        hop_length: int) -> List[Dict]:
        """Convert frame indices to time in seconds"""
        time_patterns = []
        
        for p in patterns:
            time_pattern = {
                'time1': p['start1'] * hop_length / sr,
                'time2': p['start2'] * hop_length / sr,
                'duration': p['length'] * hop_length / sr,
                'similarity': p['similarity']
            }
            time_patterns.append(time_pattern)
            
        return time_patterns

    def _group_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Group similar patterns together"""
        if not patterns:
            return []
            
        # Extract pattern features for clustering
        features = np.array([[p['duration'], p['similarity']] for p in patterns])
        
        # Normalize features
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Cluster patterns
        clustering = DBSCAN(
            eps=0.5,
            min_samples=2,
            metric='euclidean'
        ).fit(features)
        
        # Group patterns by cluster
        grouped = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # Noise points
                continue
                
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(patterns[i])
        
        # Format output
        result = []
        for label, group in grouped.items():
            result.append({
                'patterns': group,
                'avg_duration': np.mean([p['duration'] for p in group]),
                'avg_similarity': np.mean([p['similarity'] for p in group]),
                'count': len(group)
            })
        
        return result

    def analyze_matches(self,
                       matches: List[Dict],
                       features: np.ndarray,
                       sr: int = 44100,
                       hop_length: int = 512) -> List[Dict]:
        """
        Analyze matches to find repeated patterns
        
        Args:
            matches: List of potential matches
            features: Feature matrix used for matching
            
        Returns:
            Enhanced matches with pattern information
        """
        try:
            # Find repeated patterns
            patterns = self.find_repetitions(features, sr, hop_length)
            
            # Enhance matches with pattern information
            enhanced_matches = []
            for match in matches:
                # Find overlapping patterns
                match_patterns = self._find_overlapping_patterns(
                    match['start_time'],
                    match['duration'],
                    patterns
                )
                
                # Add pattern information to match
                enhanced_match = match.copy()
                enhanced_match['patterns'] = match_patterns
                enhanced_match['pattern_count'] = len(match_patterns)
                
                if match_patterns:
                    # Boost confidence if match is part of a pattern
                    enhanced_match['confidence'] *= (1 + 0.1 * len(match_patterns))
                
                enhanced_matches.append(enhanced_match)
            
            logger.info("Enhanced matches with pattern information")
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error analyzing matches: {str(e)}")
            raise

    def _find_overlapping_patterns(self,
                                 start_time: float,
                                 duration: float,
                                 patterns: List[Dict]) -> List[Dict]:
        """Find patterns that overlap with given time range"""
        overlapping = []
        
        for pattern_group in patterns:
            for pattern in pattern_group['patterns']:
                # Check both occurrences of the pattern
                for time in [pattern['time1'], pattern['time2']]:
                    if (time <= start_time + duration and
                        time + pattern['duration'] >= start_time):
                        overlapping.append(pattern)
                        break  # Only count each pattern once
        
        return overlapping
