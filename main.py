import gradio as gr
import logging
import librosa
import numpy as np
import soundfile as sf
import os
from tempfile import NamedTemporaryFile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_analyze_audio(audio_path):
    """Helper function to load and analyze a single audio file"""
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "Duration": f"{duration:.2f} seconds",
        "Sample Rate": f"{sr} Hz",
        "Estimated Tempo": f"{tempo:.0f} BPM",
        "Number of Samples": len(y)
    }

def analyze_audio(sample_audio, comparison_audio):
    """Analyze basic properties of the uploaded audio files"""
    logger.info("Analyze audio function called")
    logger.info(f"Sample audio path: {sample_audio}")
    logger.info(f"Comparison audio path: {comparison_audio}")
    
    try:
        # Analyze both audio files
        sample_analysis = load_and_analyze_audio(sample_audio)
        comp_analysis = load_and_analyze_audio(comparison_audio)
        
        # Create analysis results
        results = {
            "Sample Audio": sample_analysis,
            "Comparison Audio": comp_analysis
        }
        
        logger.info("Analysis completed successfully")
        logger.info(f"Results: {results}")
        
        return results, sample_audio, comparison_audio
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}, None, None

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_audio,
    inputs=[
        gr.Audio(label="Upload Sample Audio", type="filepath"),
        gr.Audio(label="Upload Comparison Audio", type="filepath")
    ],
    outputs=[
        gr.JSON(label="Audio Analysis"),
        gr.Audio(label="Sample Audio Playback"),
        gr.Audio(label="Comparison Audio Playback")
    ],
    title="Audio Analysis Demo",
    description="Upload two audio files to analyze their properties"
)

# For Lightning AI
app = demo.app

if __name__ == "__main__":
    logger.info("Starting application")
    demo.launch()
