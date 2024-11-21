import numpy as np
import logging
from typing import Dict, Any, Optional
from .base_analyzer import BaseAnalyzer
from basic_pitch.inference import predict

logger = logging.getLogger(__name__)

class MIDIAnalyzer(BaseAnalyzer):
    def __init__(self):
        """Initialize the MIDI analyzer."""
        super().__init__()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            import tensorflow as tf
            self.tf_available = True
            logger.info("TensorFlow is available for MIDI analysis")
        except ImportError:
            self.tf_available = False
            logger.warning("TensorFlow not available. MIDI analysis will be limited.")

    def analyze(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
        """Analyze audio and extract MIDI features."""
        if not isinstance(audio, np.ndarray):
            logger.error("Input audio must be a numpy array")
            return self._create_empty_result()

        try:
            return self._analyze_impl(audio, sr)
        except Exception as e:
            logger.error(f"Error in MIDI analysis: {str(e)}")
            return self._create_empty_result()

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result structure."""
        return {
            'midi_data': None,
            'note_events': [],
            'metadata': {
                'num_notes': 0,
                'duration': 0,
                'pitch_range': {
                    'min': 0,
                    'max': 0
                },
                'error': 'Analysis failed or dependencies missing'
            }
        }

    def _analyze_impl(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Internal implementation of MIDI analysis."""
        if not self.tf_available:
            logger.warning("TensorFlow not available - returning empty MIDI analysis")
            return self._create_empty_result()

        # Ensure audio is the right shape
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)

        try:
            # Run Basic Pitch prediction
            midi_data, notes, _ = predict(audio_data, sr)
            
            # Extract relevant features
            note_events = []
            for note in notes:
                note_event = {
                    'pitch': note.pitch,
                    'start_time': note.start_time,
                    'end_time': note.end_time,
                    'velocity': note.velocity,
                    'pitch_bend': note.pitch_bend
                }
                note_events.append(note_event)
            
            # Create feature dictionary
            features = {
                'midi_data': midi_data,
                'note_events': note_events,
                'metadata': {
                    'num_notes': len(note_events),
                    'duration': note_events[-1]['end_time'] if note_events else 0,
                    'pitch_range': {
                        'min': min(n['pitch'] for n in note_events) if note_events else 0,
                        'max': max(n['pitch'] for n in note_events) if note_events else 0
                    }
                }
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error in MIDI prediction: {str(e)}")
            return self._create_empty_result()

    def compare_midi(self, midi1: Dict[str, Any], midi2: Dict[str, Any], tolerance: float = 0.1) -> float:
        """
        Compare two MIDI feature sets for similarity.
        
        Args:
            midi1 (dict): First MIDI feature set
            midi2 (dict): Second MIDI feature set
            tolerance (float): Time tolerance for note matching
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not midi1['note_events'] or not midi2['note_events']:
            return 0.0
            
        matches = 0
        total_notes = len(midi1['note_events'])
        
        for note1 in midi1['note_events']:
            for note2 in midi2['note_events']:
                # Check if notes match within tolerance
                pitch_match = note1['pitch'] == note2['pitch']
                time_match = abs(note1['start_time'] - note2['start_time']) <= tolerance
                
                if pitch_match and time_match:
                    matches += 1
                    break
                    
        return matches / total_notes if total_notes > 0 else 0.0
