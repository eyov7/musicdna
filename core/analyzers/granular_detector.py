import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple
from .base_analyzer import BaseAnalyzer
from .spectral_analyzer import SpectralAnalyzer
from .stem_analyzer import StemAnalyzer
from .midi_analyzer import MIDIAnalyzer

logger = logging.getLogger(__name__)

class GranularSampleDetector(BaseAnalyzer):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.spectral_analyzer = SpectralAnalyzer()
        self.stem_analyzer = StemAnalyzer(device=device)
        self.midi_analyzer = MIDIAnalyzer(device=device)
        logger.info(f"Initialized Granular Sample Detector on device: {device}")
        
    def analyze_sample(self, audio_data: np.ndarray, sr: int = 22050) -> Dict:
        """
        Perform multi-level analysis on an audio sample.
        
        Args:
            audio_data: Audio data to analyze
            sr: Sample rate
            
        Returns:
            Dictionary containing analysis results at all levels
        """
        try:
            # Level 1: Full Spectral Analysis
            spectral_features = self.spectral_analyzer.analyze(audio_data)
            
            # Level 2: Stem Analysis
            stem_features = self.stem_analyzer.analyze(audio_data)
            
            # Level 3: MIDI Analysis per stem
            midi_features = {}
            for stem_name, stem_audio in stem_features['stems'].items():
                midi_features[stem_name] = self.midi_analyzer.analyze(stem_audio, sr)
            
            return {
                'full_spectral': spectral_features,
                'stems': stem_features,
                'midi_per_stem': midi_features,
                'metadata': {
                    'duration': len(audio_data) / sr,
                    'sample_rate': sr
                }
            }
            
        except Exception as e:
            logger.error(f"Error in granular analysis: {str(e)}")
            return {'error': str(e)}
            
    def sliding_windows(self, audio_data: np.ndarray, window_size: float = 5.0, 
                       hop_size: float = 2.5, sr: int = 22050) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generate sliding windows over the audio data.
        
        Args:
            audio_data: Audio data to window
            window_size: Window size in seconds
            hop_size: Hop size in seconds
            sr: Sample rate
            
        Returns:
            List of tuples containing (window_data, start_time, end_time)
        """
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        windows = []
        
        for start in range(0, len(audio_data) - window_samples, hop_samples):
            end = start + window_samples
            window = audio_data[start:end]
            start_time = start / sr
            end_time = end / sr
            windows.append((window, start_time, end_time))
            
        return windows
        
    def find_in_track(self, sample_analysis: Dict, track_data: np.ndarray, 
                     sr: int = 22050, threshold: float = 0.8) -> List[Dict]:
        """
        Find occurrences of a sample in a full track using sliding window analysis.
        
        Args:
            sample_analysis: Analysis data from analyze_sample()
            track_data: Audio data of the full track
            sr: Sample rate
            threshold: Confidence threshold for matches
            
        Returns:
            List of dictionaries containing match information
        """
        matches = []
        windows = self.sliding_windows(track_data, sr=sr)
        
        for window_data, start_time, end_time in windows:
            window_analysis = self.analyze_sample(window_data, sr)
            
            match_data = {
                'time_start': start_time,
                'time_end': end_time,
                'confidence_per_stem': {},
                'midi_matches': {}
            }
            
            # Compare each stem
            for stem_name in ['drums', 'bass', 'vocals', 'other']:
                # Spectral confidence
                spec_confidence = self.compare_spectrograms(
                    sample_analysis['stems']['stem_features'][stem_name],
                    window_analysis['stems']['stem_features'][stem_name]
                )
                
                # MIDI confidence
                midi_confidence = self.midi_analyzer.compare_midi(
                    sample_analysis['midi_per_stem'][stem_name],
                    window_analysis['midi_per_stem'][stem_name]
                )
                
                match_data['confidence_per_stem'][stem_name] = spec_confidence
                match_data['midi_matches'][stem_name] = midi_confidence
            
            # Calculate overall confidence
            overall_confidence = self.calculate_overall_confidence(match_data)
            match_data['overall_confidence'] = overall_confidence
            
            if overall_confidence >= threshold:
                matches.append(match_data)
        
        return matches
        
    def compare_spectrograms(self, spec1: Dict, spec2: Dict) -> float:
        """
        Compare two sets of spectral features.
        
        Args:
            spec1: First set of spectral features
            spec2: Second set of spectral features
            
        Returns:
            Similarity score between 0 and 1
        """
        # Compare mel spectrograms
        mel_sim = np.mean(np.abs(spec1['mel_spectrogram'] - spec2['mel_spectrogram']))
        
        # Compare MFCCs
        mfcc_sim = np.mean(np.abs(spec1['mfcc'] - spec2['mfcc']))
        
        # Compare chroma features
        chroma_sim = np.mean(np.abs(spec1['chroma'] - spec2['chroma']))
        
        # Weight and combine similarities
        weights = {'mel': 0.4, 'mfcc': 0.4, 'chroma': 0.2}
        total_sim = (weights['mel'] * mel_sim + 
                    weights['mfcc'] * mfcc_sim + 
                    weights['chroma'] * chroma_sim)
                    
        return 1 - total_sim  # Convert distance to similarity
        
    def calculate_overall_confidence(self, match_data: Dict) -> float:
        """
        Calculate overall confidence from stem and MIDI confidences.
        
        Args:
            match_data: Dictionary containing confidence scores
            
        Returns:
            Overall confidence score between 0 and 1
        """
        stem_weights = {
            'drums': 0.3,
            'bass': 0.3,
            'vocals': 0.2,
            'other': 0.2
        }
        
        # Weight spectral and MIDI confidences
        spectral_weight = 0.6
        midi_weight = 0.4
        
        overall = 0.0
        for stem_name, weight in stem_weights.items():
            stem_conf = (
                spectral_weight * match_data['confidence_per_stem'][stem_name] +
                midi_weight * match_data['midi_matches'][stem_name]
            )
            overall += weight * stem_conf
            
        return overall
