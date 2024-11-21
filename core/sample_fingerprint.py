"""
Core sample fingerprinting and detection module.
Focuses on precise sample identification through efficient fingerprinting.
"""
import numpy as np
import librosa
from typing import Tuple, List, Dict
from basic_pitch.inference import predict_and_save

class SampleFingerprint:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.fingerprint_resolution = 2048  # FFT window size
        
    def create_fingerprint(self, audio_data: np.ndarray) -> Dict:
        """
        Creates a precise fingerprint of a sample focusing on:
        - Frequency components (FFT)
        - Timing information
        - MIDI note data if melodic
        """
        # Get precise frequency components through FFT
        fft_data = np.fft.fft(audio_data, n=self.fingerprint_resolution)
        magnitude = np.abs(fft_data)[:self.fingerprint_resolution // 2]
        
        # Get timing information
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
        
        # Convert to MIDI if melodic content detected
        if self._is_melodic(audio_data):
            midi_data = self._extract_midi(audio_data)
        else:
            midi_data = None
            
        return {
            'frequency_components': magnitude,
            'onset_points': onset_frames,
            'midi_data': midi_data,
            'length': len(audio_data)
        }
    
    def _is_melodic(self, audio_data: np.ndarray) -> bool:
        """Check if sample contains melodic content worth MIDI conversion."""
        # Quick check for pitched content using spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        return np.mean(flatness) < 0.3  # Threshold for "melodicness"
    
    def _extract_midi(self, audio_data: np.ndarray) -> List:
        """Extract MIDI note data for melodic samples."""
        model_output = predict_and_save(audio_data, self.sample_rate)
        return model_output['pitch_list']  # List of (time, pitch, confidence)
    
    def find_in_track(self, sample_fp: Dict, track_data: np.ndarray, 
                     threshold: float = 0.85) -> List[Tuple[int, float]]:
        """
        Find sample occurrences in a track using sliding window FFT comparison.
        Returns list of (start_point, confidence).
        """
        window_size = len(sample_fp['frequency_components'])
        matches = []
        
        # Slide through track with overlap
        hop_length = window_size // 4
        for i in range(0, len(track_data) - window_size, hop_length):
            window = track_data[i:i + window_size]
            confidence = self._compare_window(sample_fp, window)
            
            if confidence > threshold:
                matches.append((i, confidence))
                
        return self._merge_adjacent_matches(matches)
    
    def _compare_window(self, sample_fp: Dict, window: np.ndarray) -> float:
        """Compare a window of track data with sample fingerprint."""
        # Get frequency components of window
        window_fft = np.fft.fft(window, n=self.fingerprint_resolution)
        window_magnitude = np.abs(window_fft)[:self.fingerprint_resolution // 2]
        
        # Compare frequency components
        freq_similarity = np.corrcoef(sample_fp['frequency_components'], 
                                    window_magnitude)[0, 1]
        
        # If MIDI data exists, check for melodic similarity
        if sample_fp['midi_data'] is not None:
            midi_similarity = self._compare_midi(sample_fp['midi_data'], window)
            return (freq_similarity + midi_similarity) / 2
        
        return freq_similarity
    
    def _compare_midi(self, sample_midi: List, window: np.ndarray) -> float:
        """Compare MIDI patterns between sample and window."""
        window_midi = self._extract_midi(window)
        # Compare MIDI note sequences for similarity
        # This is a simplified comparison - could be more sophisticated
        if not window_midi:
            return 0.0
        
        matches = 0
        total = len(sample_midi)
        for note in sample_midi:
            if note in window_midi:
                matches += 1
        
        return matches / total
    
    def _merge_adjacent_matches(self, matches: List[Tuple[int, float]], 
                              max_gap: int = 4410) -> List[Tuple[int, float]]:
        """Merge matches that are likely the same occurrence."""
        if not matches:
            return []
            
        merged = []
        current_start, current_conf = matches[0]
        
        for start, conf in matches[1:]:
            if start - (current_start + self.fingerprint_resolution) <= max_gap:
                # Merge by taking highest confidence
                current_conf = max(current_conf, conf)
            else:
                merged.append((current_start, current_conf))
                current_start, current_conf = start, conf
                
        merged.append((current_start, current_conf))
        return merged
