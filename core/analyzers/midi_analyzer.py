import numpy as np
import basic_pitch
from basic_pitch.inference import predict
import logging
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class MIDIAnalyzer(BaseAnalyzer):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        logger.info(f"Initialized MIDI Analyzer on device: {device}")

    def analyze(self, audio_data, sr=22050):
        """
        Analyze audio data and extract MIDI information using Basic Pitch.
        
        Args:
            audio_data (np.ndarray): Audio data to analyze
            sr (int): Sample rate of the audio
            
        Returns:
            dict: Dictionary containing MIDI features and metadata
        """
        try:
            # Get MIDI predictions
            midi_data, notes, _ = predict(audio_data, sr)
            
            # Extract key features from the MIDI data
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
            logger.error(f"Error in MIDI analysis: {str(e)}")
            return {
                'midi_data': None,
                'note_events': [],
                'metadata': {'error': str(e)}
            }

    def compare_midi(self, midi1, midi2, tolerance=0.1):
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
