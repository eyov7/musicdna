"""
Stem-based sample detection system.
Uses isolated stems as natural audio decomposition for precise sample matching.
"""
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
from typing import Dict, List, Tuple

class StemMatcher:
    def __init__(self):
        # Load Demucs model for stem separation
        self.separator = get_model('htdemucs')
        self.stem_types = ['drums', 'bass', 'vocals', 'other']
        
    def separate_stems(self, audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, np.ndarray]:
        """Separate audio into stems."""
        # Convert to torch tensor and reshape for Demucs
        audio_tensor = torch.tensor(audio).reshape(1, -1)
        
        # Separate stems
        stems = apply_model(self.separator, audio_tensor, sample_rate)[0]
        return {
            name: stem.numpy()
            for name, stem in zip(self.stem_types, stems)
        }
    
    def classify_sample(self, sample: np.ndarray, sample_rate: int = 44100) -> str:
        """Determine which stem type a sample belongs to."""
        # Separate sample into stems
        stems = self.separate_stems(sample, sample_rate)
        
        # Calculate energy in each stem
        energies = {
            name: np.sum(np.abs(stem))
            for name, stem in stems.items()
        }
        
        # Return stem type with highest energy
        return max(energies.items(), key=lambda x: x[1])[0]
    
    def find_sample_in_stems(self, sample: np.ndarray, 
                           stems: Dict[str, np.ndarray],
                           threshold: float = 0.8) -> List[Dict]:
        """
        Find sample occurrences in appropriate stem.
        Returns list of matches with timing and confidence.
        """
        # Determine which stem to search in
        target_stem = self.classify_sample(sample)
        
        # Search in the appropriate stem
        matches = self._find_in_stem(sample, stems[target_stem], threshold)
        
        return [{
            'stem': target_stem,
            'start_time': start,
            'confidence': conf
        } for start, conf in matches]
    
    def _find_in_stem(self, sample: np.ndarray, stem: np.ndarray, 
                     threshold: float) -> List[Tuple[int, float]]:
        """Find sample occurrences in a specific stem."""
        sample_len = len(sample)
        matches = []
        
        # Normalize sample and prepare correlation
        sample_normalized = sample / np.sqrt(np.sum(sample ** 2))
        
        # Slide through stem with overlap
        hop_length = sample_len // 2
        for i in range(0, len(stem) - sample_len, hop_length):
            window = stem[i:i + sample_len]
            
            # Normalize window
            window_normalized = window / np.sqrt(np.sum(window ** 2))
            
            # Calculate correlation
            correlation = np.sum(sample_normalized * window_normalized)
            
            if correlation > threshold:
                matches.append((i, correlation))
        
        return self._merge_matches(matches)
    
    def _merge_matches(self, matches: List[Tuple[int, float]], 
                      max_gap: int = 4410) -> List[Tuple[int, float]]:
        """Merge nearby matches."""
        if not matches:
            return []
            
        merged = []
        current_start, current_conf = matches[0]
        
        for start, conf in matches[1:]:
            if start - current_start <= max_gap:
                # Keep the higher confidence match
                if conf > current_conf:
                    current_conf = conf
            else:
                merged.append((current_start, current_conf))
                current_start, current_conf = start, conf
        
        merged.append((current_start, current_conf))
        return merged
    
    def analyze_multi_track(self, tracks: List[np.ndarray], 
                          sample: np.ndarray) -> List[Dict]:
        """Analyze multiple tracks for sample occurrences."""
        all_matches = []
        
        for track_idx, track in enumerate(tracks):
            # Separate track into stems
            stems = self.separate_stems(track)
            
            # Find matches in stems
            matches = self.find_sample_in_stems(sample, stems)
            
            # Add track information to matches
            for match in matches:
                match['track_index'] = track_idx
                
            all_matches.extend(matches)
            
        return sorted(all_matches, key=lambda x: x['confidence'], reverse=True)
