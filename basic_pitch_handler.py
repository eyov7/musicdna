from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict_and_save
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class BasicPitchConverter:
    def __init__(self):
        logger.info("Initializing Basic Pitch converter")
        self.model_path = ICASSP_2022_MODEL_PATH
        logger.info("Basic Pitch initialized successfully")

    def convert_to_midi(self, audio_path):
        """Convert audio to MIDI using Basic Pitch"""
        logger.info(f"Converting audio to MIDI: {audio_path}")
        try:
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                midi_path = temp_dir / "output.midi"
                
                # Convert to MIDI
                predict_and_save(
                    [audio_path],
                    str(temp_dir),
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=False
                )
                
                # Read MIDI file
                with open(midi_path, 'rb') as f:
                    midi_data = f.read()
                
            logger.info("MIDI conversion completed successfully")
            return midi_data
            
        except Exception as e:
            logger.error(f"Error in MIDI conversion: {str(e)}")
            raise
