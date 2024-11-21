import numpy as np
import librosa
import gradio as gr
import matplotlib.pyplot as plt
import os
from tempfile import NamedTemporaryFile
from demucs_handler import DemucsProcessor
from basic_pitch_handler import BasicPitchConverter
from similarity_detector import SimilarityDetector
from visualization import AudioVisualizer

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
demucs = DemucsProcessor()
basic_pitch = BasicPitchConverter()
detector = SimilarityDetector()
visualizer = AudioVisualizer()

def process_audio(sample_path: str, song_path: str) -> dict:
    """Process audio files and return matches with visualizations"""
    try:
        # Load and process audio files
        sample_y, sr = librosa.load(sample_path)
        song_y, _ = librosa.load(song_path, sr=sr)
        
        # Get stems for both
        sample_stems, _ = demucs.separate_stems(sample_path)
        song_stems, _ = demucs.separate_stems(song_path)
        
        # Detect matches
        results = detector.detect_matches(sample_y, song_y)
        matches = results.get('matches', [])
        
        # Format output
        output_text = "Found matches:\n"
        for match in matches:
            output_text += f"Time: {match['start_time']:.2f}s, Duration: {match['duration']:.2f}s, Confidence: {match['confidence']:.2%}\n"
        
        return output_text
        
    except Exception as e:
        logger.error(f"Error in sample analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"

def create_interface():
    """Create Gradio interface"""
    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(label="Upload Sample (5-15 seconds)", type="filepath"),
            gr.Audio(label="Upload Full Song", type="filepath")
        ],
        outputs=[
            gr.Textbox(label="Detection Results", lines=10)
        ],
        title="MusicDNA - Sample Detection",
        description="""## Advanced Sample Detection System
        
        Upload a sample audio clip and a full song to find where the sample appears in the song.
        
        Instructions:
        1. Upload a short sample (5-15 seconds)
        2. Upload the full song to analyze
        3. Click submit to find matches
        
        The system will analyze the audio and show where the sample appears in the song.
        """
    )
    return interface

# Create Gradio interface
demo = create_interface()

# For Lightning AI
app = demo.app

if __name__ == "__main__":
    demo.launch(server_port=7860)
