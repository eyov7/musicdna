import gradio as gr
import logging
import librosa
import numpy as np
import soundfile as sf
import os
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_chromagram(y, sr):
    """Create and plot chromagram for audio"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar(label='Magnitude')
    plt.title('Chromagram')
    
    # Save plot to temporary file
    with NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        plt.savefig(temp_file.name)
        plt.close()
        return temp_file.name

def compute_similarity(chroma1, chroma2):
    """Compute similarity between two chromagrams"""
    # Normalize chromagrams
    norm_chroma1 = librosa.util.normalize(chroma1, axis=0)
    norm_chroma2 = librosa.util.normalize(chroma2, axis=0)
    
    # Compute cross-similarity matrix
    similarity = np.dot(norm_chroma1.T, norm_chroma2)
    return np.mean(similarity)

def load_and_analyze_audio(audio_path):
    """Helper function to load and analyze a single audio file"""
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chromagram_path = create_chromagram(y, sr)
        
        # Convert numpy values to Python native types
        duration_float = float(duration)
        tempo_float = float(tempo)
        samples_int = int(len(y))
        
        return {
            "audio_data": (y, sr),
            "chroma": chroma,
            "chromagram_path": chromagram_path,
            "stats": {
                "Duration": f"{duration_float:.2f} seconds",
                "Sample Rate": f"{sr} Hz",
                "Estimated Tempo": f"{tempo_float:.0f} BPM",
                "Number of Samples": samples_int
            }
        }
    except Exception as e:
        logger.error(f"Error in load_and_analyze_audio: {str(e)}")
        raise

def analyze_audio(sample_audio, comparison_audio):
    """Analyze basic properties of the uploaded audio files"""
    logger.info("Analyze audio function called")
    logger.info(f"Sample audio path: {sample_audio}")
    logger.info(f"Comparison audio path: {comparison_audio}")
    
    try:
        # Analyze both audio files
        sample_analysis = load_and_analyze_audio(sample_audio)
        comp_analysis = load_and_analyze_audio(comparison_audio)
        
        # Compute similarity
        similarity_score = compute_similarity(
            sample_analysis["chroma"],
            comp_analysis["chroma"]
        )
        
        # Create analysis results
        results = {
            "Sample Audio": sample_analysis["stats"],
            "Comparison Audio": comp_analysis["stats"],
            "Similarity Score": f"{float(similarity_score):.2%}"
        }
        
        logger.info("Analysis completed successfully")
        logger.info(f"Results: {results}")
        
        return (
            results, 
            sample_audio,
            comparison_audio,
            sample_analysis["chromagram_path"],
            comp_analysis["chromagram_path"]
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}, None, None, None, None

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
        gr.Audio(label="Comparison Audio Playback"),
        gr.Image(label="Sample Chromagram"),
        gr.Image(label="Comparison Chromagram")
    ],
    title="Audio Analysis Demo",
    description="Upload two audio files to analyze their properties and similarity"
)

# For Lightning AI
app = demo.app

if __name__ == "__main__":
    logger.info("Starting application")
    demo.launch()
