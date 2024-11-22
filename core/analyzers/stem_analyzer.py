"""
Core stem analyzer implementing the natural decomposition approach.
Uses stems as primary decomposition method, with dual fingerprinting (Visual + MIDI).
"""
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from .base_analyzer import BaseAnalyzer
import librosa
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from basic_pitch import predict

logger = logging.getLogger(__name__)

class StemAnalyzer(BaseAnalyzer):
    def __init__(self, sample_rate: int = 44100):
        """Initialize the stem analyzer with models and configurations."""
        super().__init__()
        self.sample_rate = sample_rate
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize Demucs model with GPU support if available."""
        try:
            self.demucs_model = get_model('htdemucs')
            self.demucs_model.eval()
            if torch.cuda.is_available():
                self.demucs_model.cuda()
                logger.info("Using GPU for Demucs model")
            else:
                logger.info("Using CPU for Demucs model")
                
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise RuntimeError("Failed to initialize required models")

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Implement the abstract analyze method.
        Performs full stem-based analysis of the audio.
        """
        try:
            # Separate into stems
            stems = self.separate_stems(audio)
            
            # Analyze each stem
            analysis = {}
            for stem_name, stem_audio in stems.items():
                analysis[stem_name] = {
                    'fingerprint': self.create_fingerprint(stem_audio),
                    'is_melodic': self._is_melodic(stem_audio)
                }
            
            return {
                'stems': analysis,
                'primary_stem': max(
                    stems.items(),
                    key=lambda x: np.sum(np.abs(x[1]))
                )[0]
            }
            
        except Exception as e:
            logger.error(f"Error in stem analysis: {str(e)}")
            raise

    def create_fingerprint(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Create dual fingerprint (Visual + MIDI) of audio.
        """
        # Ensure correct audio format
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
            
        # Generate Visual DNA (spectrogram)
        visual_dna = self._create_visual_dna(audio)
        
        # Generate MIDI DNA if melodic content detected
        if self._is_melodic(audio):
            midi_dna = self._create_midi_dna(audio)
        else:
            midi_dna = None
            
        return {
            'visual_dna': visual_dna,
            'midi_dna': midi_dna,
            'length': audio.shape[-1]
        }
    
    def _create_visual_dna(self, audio: np.ndarray) -> np.ndarray:
        """Generate spectrogram fingerprint."""
        return librosa.feature.melspectrogram(
            y=audio[0], 
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
    
    def _create_midi_dna(self, audio: np.ndarray) -> List:
        """Generate MIDI note pattern fingerprint."""
        # Get model predictions
        model_output = predict(audio[0], self.sample_rate)
        frequencies, confidence, onset = model_output
        
        # Convert to note list format
        notes = []
        for t in range(len(frequencies)):
            for f in range(len(frequencies[t])):
                if onset[t, f] > 0.5:  # Note onset detected
                    notes.append({
                        'time': t * 0.01,  # Basic Pitch uses 10ms frames
                        'pitch': f,
                        'confidence': confidence[t, f]
                    })
        return notes
    
    def _is_melodic(self, audio: np.ndarray) -> bool:
        """Determine if audio contains significant melodic content."""
        flatness = librosa.feature.spectral_flatness(y=audio[0])[0]
        return np.mean(flatness) < 0.3
    
    def separate_stems(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose audio into stems (our natural decomposition).
        Returns dict of stems: drums, bass, vocals, other
        """
        # Prepare audio for Demucs
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        audio_tensor = torch.tensor(audio)
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
            
        # Separate stems using apply_model
        with torch.no_grad():
            stems = apply_model(self.demucs_model, audio_tensor, split=True)[0]
        
        # Convert back to numpy and create dict
        stem_names = ['drums', 'bass', 'vocals', 'other']
        return {
            name: stem.cpu().numpy() 
            for name, stem in zip(stem_names, stems)
        }
    
    def find_sample(self, sample: np.ndarray, 
                   stems: Dict[str, np.ndarray],
                   threshold: float = 0.85) -> List[Dict]:
        """
        Find sample occurrences in stems using dual fingerprinting.
        """
        # Create sample fingerprint
        sample_fp = self.create_fingerprint(sample)
        
        # Determine primary stem type
        sample_stems = self.separate_stems(sample)
        primary_stem = max(
            sample_stems.items(),
            key=lambda x: np.sum(np.abs(x[1]))
        )[0]
        
        # Search in primary stem
        matches = []
        target_stem = stems[primary_stem]
        
        # Slide through stem with overlap
        window_size = sample_fp['length']
        hop_length = window_size // 2
        
        for i in range(0, len(target_stem) - window_size, hop_length):
            window = target_stem[:, i:i + window_size]
            window_fp = self.create_fingerprint(window)
            
            # Compare fingerprints
            confidence = self._compare_fingerprints(sample_fp, window_fp)
            
            if confidence > threshold:
                matches.append({
                    'start': i,
                    'confidence': confidence,
                    'stem': primary_stem
                })
        
        return self._merge_matches(matches)
    
    def _compare_fingerprints(self, fp1: Dict, fp2: Dict) -> float:
        """Compare two fingerprints using both Visual and MIDI DNA."""
        # Compare spectrograms
        visual_sim = np.corrcoef(
            fp1['visual_dna'].flatten(),
            fp2['visual_dna'].flatten()
        )[0, 1]
        
        # If both have MIDI data, include MIDI similarity
        if fp1['midi_dna'] and fp2['midi_dna']:
            midi_sim = self._compare_midi(fp1['midi_dna'], fp2['midi_dna'])
            return (visual_sim + midi_sim) / 2
            
        return visual_sim
    
    def _compare_midi(self, midi1: List, midi2: List) -> float:
        """Compare MIDI note patterns."""
        if not midi1 or not midi2:
            return 0.0
            
        matches = 0
        total = len(midi1)
        
        for note1 in midi1:
            for note2 in midi2:
                # Compare pitch and timing
                if (abs(note1['pitch'] - note2['pitch']) <= 1 and  # 1 semitone difference
                    abs(note1['time'] - note2['time']) <= 0.05):   # 50ms timing difference
                    matches += 1
                    break
                    
        return matches / total if total > 0 else 0.0
    
    def _merge_matches(self, matches: List[Dict], 
                      max_gap: int = 4410) -> List[Dict]:
        """Merge nearby matches."""
        if not matches:
            return []
            
        merged = []
        current = matches[0]
        
        for match in matches[1:]:
            if match['start'] - (current['start'] + current['length']) <= max_gap:
                if match['confidence'] > current['confidence']:
                    current = match
            else:
                merged.append(current)
                current = match
                
        merged.append(current)
        return merged
