import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from .base_analyzer import BaseAnalyzer
from .spectral_analyzer import SpectralAnalyzer
from .stem_analyzer import StemAnalyzer
from .midi_analyzer import MIDIAnalyzer

logger = logging.getLogger(__name__)

class GranularSampleDetector(BaseAnalyzer):
    def __init__(self, device="cpu", window_size: float = 5.0, hop_size: float = 2.5):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.hop_size = hop_size
        
        # Initialize analyzers
        self.spectral_analyzer = SpectralAnalyzer()
        self.stem_analyzer = StemAnalyzer(device=device)
        self.midi_analyzer = MIDIAnalyzer(device=device)
        
        logger.info(f"Initialized Granular Sample Detector on {device}")
        logger.info(f"Window size: {window_size}s, Hop size: {hop_size}s")
        
    def create_spectrogram(self, audio_data: np.ndarray) -> Dict:
        """Create spectrogram with enhanced feature extraction"""
        specs = self.spectral_analyzer.analyze(audio_data)
        
        # Add additional features for better matching
        if specs:
            specs['onset_strength'] = librosa.onset.onset_strength(y=audio_data)
            specs['tempogram'] = librosa.feature.tempogram(onset_envelope=specs['onset_strength'])
        return specs
        
    def analyze_sample(self, audio_data: np.ndarray, sr: int = 22050) -> Dict:
        """
        Multi-level analysis of an audio sample.
        
        Args:
            audio_data: Audio data to analyze
            sr: Sample rate
            
        Returns:
            Dictionary containing analysis at multiple levels
        """
        try:
            # Level 1: Full Audio Analysis
            full_analysis = {
                'spectral': self.create_spectrogram(audio_data),
                'duration': len(audio_data) / sr,
                'rms': np.sqrt(np.mean(audio_data**2))
            }
            
            # Level 2: Stem Analysis
            stem_data = self.stem_analyzer.analyze(audio_data, sr)
            stems = stem_data['stems']
            stem_analysis = {}
            
            # Analyze each stem
            for stem_name, stem_audio in stems.items():
                stem_analysis[stem_name] = {
                    'spectral': self.create_spectrogram(stem_audio),
                    'midi': self.midi_analyzer.analyze(stem_audio, sr),
                    'features': {
                        'rms': np.sqrt(np.mean(stem_audio**2)),
                        'peak': np.max(np.abs(stem_audio)),
                        'zero_crossings': np.sum(np.abs(np.diff(np.signbit(stem_audio))))
                    }
                }
            
            return {
                'full': full_analysis,
                'stems': stem_analysis,
                'metadata': {
                    'sample_rate': sr,
                    'duration': len(audio_data) / sr
                }
            }
            
        except Exception as e:
            logger.error(f"Error in sample analysis: {str(e)}")
            return {'error': str(e)}
            
    def sliding_windows(self, audio_data: np.ndarray, sr: int = 22050) -> List[Tuple[np.ndarray, float, float]]:
        """Generate sliding windows with overlap"""
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)
        windows = []
        
        for start in range(0, len(audio_data) - window_samples, hop_samples):
            end = start + window_samples
            window = audio_data[start:end]
            # Apply windowing function for smooth transitions
            window = window * np.hanning(len(window))
            windows.append((window, start / sr, end / sr))
            
        return windows
        
    def find_in_track(self, sample_analysis: Dict, track_data: np.ndarray, 
                     sr: int = 22050, threshold: float = 0.8) -> List[Dict]:
        """
        Find sample occurrences in full track with detailed confidence scoring
        """
        matches = []
        windows = self.sliding_windows(track_data, sr)
        
        for window_data, start_time, end_time in windows:
            window_analysis = self.analyze_sample(window_data, sr)
            
            match_data = {
                'time_start': start_time,
                'time_end': end_time,
                'confidence': {},
                'transformations': {}
            }
            
            # Compare at each level
            for stem_name in ['drums', 'bass', 'vocals', 'other']:
                stem_confidence = self._calculate_stem_confidence(
                    sample_analysis['stems'][stem_name],
                    window_analysis['stems'][stem_name]
                )
                
                match_data['confidence'][stem_name] = stem_confidence
                
                # Detect potential transformations
                if stem_confidence['total'] > threshold:
                    transformations = self._detect_transformations(
                        sample_analysis['stems'][stem_name],
                        window_analysis['stems'][stem_name]
                    )
                    match_data['transformations'][stem_name] = transformations
            
            # Calculate overall confidence
            match_data['total_confidence'] = self._calculate_total_confidence(match_data['confidence'])
            
            if match_data['total_confidence'] >= threshold:
                matches.append(match_data)
        
        return matches
        
    def _calculate_stem_confidence(self, sample_stem: Dict, window_stem: Dict) -> Dict:
        """Calculate detailed confidence scores for a stem"""
        # Spectral similarity (50%)
        spectral_conf = self.compare_spectrograms(sample_stem['spectral'], window_stem['spectral'])
        
        # MIDI similarity (30%)
        midi_conf = self.midi_analyzer.compare_midi(sample_stem['midi'], window_stem['midi'])
        
        # Feature similarity (20%)
        feature_conf = self._compare_features(sample_stem['features'], window_stem['features'])
        
        total = 0.5 * spectral_conf + 0.3 * midi_conf + 0.2 * feature_conf
        
        return {
            'spectral': spectral_conf,
            'midi': midi_conf,
            'features': feature_conf,
            'total': total
        }
        
    def _detect_transformations(self, sample_stem: Dict, window_stem: Dict) -> Dict:
        """Detect potential audio transformations"""
        transformations = {}
        
        # Pitch shift detection using MIDI
        pitch_diff = self.midi_analyzer.detect_pitch_shift(
            sample_stem['midi'], window_stem['midi']
        )
        if abs(pitch_diff) > 0.5:  # More than half semitone
            transformations['pitch_shift'] = pitch_diff
            
        # Time stretch detection
        time_ratio = self._detect_time_stretch(
            sample_stem['spectral'], window_stem['spectral']
        )
        if abs(1 - time_ratio) > 0.1:  # More than 10% stretch
            transformations['time_stretch'] = time_ratio
            
        return transformations
        
    def _detect_time_stretch(self, spec1: Dict, spec2: Dict) -> float:
        """Detect time stretching factor between spectrograms"""
        tempo1 = np.mean(spec1['tempogram'])
        tempo2 = np.mean(spec2['tempogram'])
        return tempo1 / tempo2 if tempo2 != 0 else 1.0
        
    def _compare_features(self, features1: Dict, features2: Dict) -> float:
        """Compare basic audio features"""
        similarities = []
        for key in ['rms', 'peak', 'zero_crossings']:
            if key in features1 and key in features2:
                ratio = min(features1[key], features2[key]) / max(features1[key], features2[key])
                similarities.append(ratio)
        return np.mean(similarities) if similarities else 0.0
        
    def _calculate_total_confidence(self, confidence_dict: Dict) -> float:
        """Calculate overall confidence using stem weights"""
        weights = self.stem_analyzer.get_stem_weights()
        total = 0.0
        
        for stem_name, weight in weights.items():
            if stem_name in confidence_dict:
                total += weight * confidence_dict[stem_name]['total']
                
        return total
