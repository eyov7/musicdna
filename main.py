import gradio as gr
import logging
import librosa
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_audio(sample_audio, comparison_audio):
    """Analyze basic properties of the uploaded audio files"""
    logger.info("Analyze audio function called")
    logger.info(f"Sample audio: {sample_audio}")
    logger.info(f"Comparison audio: {comparison_audio}")
    
    try:
        # Load and analyze sample audio
        sample_y, sample_sr = librosa.load(sample_audio)
        sample_duration = librosa.get_duration(y=sample_y, sr=sample_sr)
        sample_tempo, _ = librosa.beat.beat_track(y=sample_y, sr=sample_sr)
        
        # Load and analyze comparison audio
        comp_y, comp_sr = librosa.load(comparison_audio)
        comp_duration = librosa.get_duration(y=comp_y, sr=comp_sr)
        comp_tempo, _ = librosa.beat.beat_track(y=comp_y, sr=comp_sr)
        
        # Create analysis results
        results = {
            "Sample Audio": {
                "Duration": f"{sample_duration:.2f} seconds",
                "Sample Rate": f"{sample_sr} Hz",
                "Estimated Tempo": f"{sample_tempo:.0f} BPM"
            },
            "Comparison Audio": {
                "Duration": f"{comp_duration:.2f} seconds",
                "Sample Rate": f"{comp_sr} Hz",
                "Estimated Tempo": f"{comp_tempo:.0f} BPM"
            }
        }
        
        logger.info(f"Analysis results: {results}")
        return results, sample_audio, comparison_audio
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}")
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
