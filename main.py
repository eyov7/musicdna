import numpy as np
import librosa
import gradio as gr
import matplotlib.pyplot as plt
import os
from tempfile import NamedTemporaryFile
from similarity_detector import SimilarityDetector

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize detector
detector = SimilarityDetector()

def process_audio(sample_path: str, song_path: str, use_advanced: bool) -> str:
    """Process audio files and return matches with visualizations"""
    try:
        # Load and process audio files
        sample_y, sr = librosa.load(sample_path, sr=22050)  # Use consistent sample rate
        song_y, _ = librosa.load(song_path, sr=22050)
        
        # Detect matches
        results = detector.detect_matches(sample_y, song_y, use_advanced=use_advanced)
        matches = results.get('matches', [])
        
        if not matches:
            return "No matches found. Try adjusting the sample or using a different section."
        
        # Format output
        output_text = f"Found matches using {'advanced' if use_advanced else 'basic'} detection:\n\n"
        for i, match in enumerate(matches, 1):
            output_text += f"{i}. Time: {match['start_time']:.2f}s\n"
            output_text += f"   Duration: {match['duration']:.2f}s\n"
            output_text += f"   Confidence: {match['confidence']:.2%}\n"
            output_text += f"   Method: {match['method']}\n\n"
        
        return output_text
        
    except Exception as e:
        logger.error(f"Error in sample analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error processing audio: {str(e)}"

def create_interface():
    """Create Gradio interface"""
    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(label="Upload Sample (5-15 seconds)", type="filepath"),
            gr.Audio(label="Upload Full Song", type="filepath"),
            gr.Checkbox(label="Use Advanced Detection", value=True, 
                       info="Enable stem separation and pitch tracking (slower but more accurate)")
        ],
        outputs=[
            gr.Textbox(label="Detection Results", lines=15)
        ],
        title="MusicDNA - Sample Detection",
        description="""
        ## Advanced Sample Detection System
        
        Upload a sample audio clip and a full song to find where the sample appears in the song.
        
        Features:
        - Basic Detection: Uses chroma features for quick matching
        - Advanced Detection: Uses stem separation and pitch tracking for more accurate results
        
        Instructions:
        1. Upload a short sample (5-15 seconds)
        2. Upload the full song to analyze
        3. Choose detection method
        4. Click submit to find matches
        
        The system will analyze the audio and show where the sample appears in the song.
        """
    )
    return interface

# Create Gradio interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
