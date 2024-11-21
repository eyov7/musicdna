import numpy as np
import librosa
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

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
                 window_overlap: float = 0.5,  # Window overlap ratio
                 context_size: int = 3,
                 base_threshold: float = 0.8,
                 n_freq_bands: int = 8):
        self.min_duration = min_duration
        self.window_overlap = window_overlap
        self.context_size = context_size
        self.mix_analyzer = MixAnalyzer(
            base_threshold=base_threshold,
            max_threshold_reduction=0.3,
            density_sensitivity=0.5
        )
        self.freq_analyzer = FrequencyAnalyzer(n_bands=n_freq_bands)
        self.pattern_analyzer = PatternAnalyzer(
            min_pattern_length=min_duration,
            max_pattern_gap=0.5,
            similarity_threshold=0.85
        )
        self.visualizer = AudioVisualizer()

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
        hop_size = int(window_length * (1 - self.window_overlap))
        
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

    def analyze_audio(self, 
                     sample_audio: np.ndarray,
                     song_audio: np.ndarray,
                     sr: int = 44100) -> tuple:
        """
        Analyze audio and compute weighted features
        """
        try:
            # Calculate frequency band weights
            weights = self.freq_analyzer.calculate_band_weights(
                sample_audio,
                reference_audio=song_audio
            )
            
            # Analyze frequency content
            sample_freqs = self.freq_analyzer.analyze_frequency_content(sample_audio)
            song_freqs = self.freq_analyzer.analyze_frequency_content(song_audio)
            
            # Apply weights
            weighted_sample = self.freq_analyzer.apply_band_weights(sample_freqs, weights)
            weighted_song = self.freq_analyzer.apply_band_weights(song_freqs, weights)
            
            return weighted_sample, weighted_song, weights
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            raise

    def detect_matches(self,
                      sample_audio: np.ndarray,
                      song_audio: np.ndarray,
                      stems: Optional[Dict[str, np.ndarray]] = None,
                      sr: int = 44100) -> List[Dict]:
        """
        Detect sample matches using multiple analysis methods
        """
        try:
            # 1. Analyze audio and get weighted features
            sample_features, song_features, freq_weights = self.analyze_audio(
                sample_audio,
                song_audio,
                sr=sr
            )
            
            # 2. Calculate mix density and dynamic threshold
            if stems is not None:
                density, threshold = self.mix_analyzer.analyze_section(stems)
            else:
                density, threshold = np.ones(len(song_features)), np.ones(len(song_features))
            
            # 3. Get raw confidence scores using weighted features
            confidence_scores = self.calculate_similarity(sample_features, song_features)
            
            # 4. Adjust confidence based on mix density
            adjusted_scores = self.mix_analyzer.adjust_confidence_scores(
                confidence_scores,
                density
            )
            
            # 5. Find initial matches using dynamic threshold
            matches = []
            for i in range(len(adjusted_scores)):
                if adjusted_scores[i] > threshold[i]:
                    match = {
                        'start_time': i * 512 / sr,
                        'confidence': float(adjusted_scores[i]),
                        'mix_density': float(density[i]),
                        'threshold': float(threshold[i]),
                        'freq_weights': freq_weights.tolist()
                    }
                    matches.append(match)
            
            # 6. Group continuous matches
            grouped_matches = self.group_continuous_matches(matches)
            
            # 7. Filter by minimum duration
            duration_filtered = [m for m in grouped_matches 
                               if m['duration'] >= self.min_duration]
            
            # 8. Enhance matches with pattern information
            final_matches = self.pattern_analyzer.analyze_matches(
                duration_filtered,
                song_features,
                sr=sr,
                hop_length=512
            )
            
            # Create visualizations
            freq_viz = self.visualizer.plot_frequency_bands(
                sample=sample_audio,
                song=song_audio,
                freq_weights=freq_weights,
                matches=final_matches
            )
            
            pattern_viz = self.visualizer.plot_pattern_map(
                song=song_audio,
                patterns=self.pattern_analyzer.get_patterns(),
                matches=final_matches
            )
            
            mix_viz = self.visualizer.plot_mix_density(
                density=density,
                matches=final_matches,
                confidence_scores=confidence_scores
            )
            
            # Sort matches by confidence
            final_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Found {len(final_matches)} matches after all analysis")
            return {
                'matches': final_matches,
                'visualizations': {
                    'frequency': freq_viz,
                    'pattern': pattern_viz,
                    'mix_density': mix_viz
                }
            }
            
        except Exception as e:
            logger.error(f"Error in match detection: {str(e)}")
            raise

    def calculate_similarity(self, 
                           sample_features: np.ndarray,
                           song_features: np.ndarray) -> np.ndarray:
        """
        Compute similarity between frequency-weighted features
        """
        try:
            # Calculate cosine similarity for each frame
            similarities = []
            for i in range(song_features.shape[1] - sample_features.shape[1] + 1):
                frame = song_features[:, i:i+sample_features.shape[1]]
                sim = np.sum(sample_features * frame) / (
                    np.sqrt(np.sum(sample_features**2)) * 
                    np.sqrt(np.sum(frame**2))
                )
                similarities.append(sim)
            
            return np.array(similarities)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise

    def group_continuous_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        Group continuous matches
        """
        grouped_matches = []
        current_match = None
        
        for match in matches:
            if current_match is None:
                current_match = match
            elif match['start_time'] - current_match['start_time'] < 1:
                current_match['duration'] = match['start_time'] - current_match['start_time']
            else:
                grouped_matches.append(current_match)
                current_match = match
        
        if current_match is not None:
            grouped_matches.append(current_match)
        
        return grouped_matches
