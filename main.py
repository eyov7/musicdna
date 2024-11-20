import gradio as gr
import logging
import librosa
import numpy as np
import soundfile as sf
import os
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from demucs_handler import DemucsProcessor
from basic_pitch_handler import BasicPitchConverter
from similarity_detector import SimilarityDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize processors
demucs = DemucsProcessor()
basic_pitch = BasicPitchConverter()
similarity_detector = SimilarityDetector(
    min_duration=1.0,     # Minimum 1 second match
    window_overlap=0.5,    # 50% window overlap
    context_size=3        # Consider 3 windows before and after
)

def create_similarity_plot(similarity_scores: np.ndarray, matches, sample_length, song_length):
    """Create visualization of similarity over time with marked matches"""
    plt.figure(figsize=(15, 6))
    
    # Plot similarity timeline
    time_axis = np.linspace(0, song_length, len(similarity_scores))
    plt.plot(time_axis, similarity_scores, label='Similarity Score', alpha=0.7)
    
    # Plot match regions
    for match in matches:
        plt.axvspan(match.start_time, match.end_time, 
                   color='green', alpha=0.3)
        plt.plot(match.start_time, 0.8, 'r^', 
                label=f'Match (conf: {match.confidence:.2f}, context: {match.context_score:.2f})')
    
    plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold')
    plt.title('Sample Detection Timeline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    with NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        return temp_file.name

def analyze_sample(sample_audio, full_song):
    """Find occurrences of sample in the full song"""
    logger.info("Starting sample analysis")
    logger.info(f"Sample audio path: {sample_audio}")
    logger.info(f"Full song path: {full_song}")
    
    try:
        # Load and process audio files
        sample_y, sr = librosa.load(sample_audio)
        song_y, _ = librosa.load(full_song, sr=sr)
        
        # Get stems for both
        sample_stems, _ = demucs.separate_stems(sample_audio)
        song_stems, _ = demucs.separate_stems(full_song)
        
        # Extract features (focusing on harmonic content from 'other' stem)
        sample_chroma = librosa.feature.chroma_cqt(
            y=librosa.to_mono(sample_stems['other']), 
            sr=sr
        )
        song_chroma = librosa.feature.chroma_cqt(
            y=librosa.to_mono(song_stems['other']), 
            sr=sr
        )
        
        # Find matches using sliding window
        matches = similarity_detector.sliding_window_similarity(
            sample_chroma,
            song_chroma,
            sr=sr
        )
        
        # Get raw similarity scores for visualization
        similarity_matrix = librosa.segment.cross_similarity(
            sample_chroma, song_chroma, mode='affinity'
        )
        similarity_scores = np.max(similarity_matrix, axis=0)
        
        # Create visualization
        sample_length = librosa.get_duration(y=sample_y, sr=sr)
        song_length = librosa.get_duration(y=song_y, sr=sr)
        plot_path = create_similarity_plot(
            similarity_scores,
            matches,
            sample_length,
            song_length
        )
        
        # Prepare results
        results = {
            "Sample Length": f"{sample_length:.2f} seconds",
            "Song Length": f"{song_length:.2f} seconds",
            "Matches Found": len(matches),
            "Detailed Matches": [
                {
                    "Start Time": f"{m.start_time:.2f}s",
                    "End Time": f"{m.end_time:.2f}s",
                    "Duration": f"{m.end_time - m.start_time:.2f}s",
                    "Confidence": f"{m.confidence:.2%}",
                    "Context Score": f"{m.context_score:.2%}"
                }
                for m in matches
            ]
        }
        
        logger.info("Analysis completed successfully")
        logger.info(f"Results: {results}")
        
        return (
            results,
            sample_audio,
            full_song,
            plot_path
        )
        
    except Exception as e:
        logger.error(f"Error in sample analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}, None, None, None

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_sample,
    inputs=[
        gr.Audio(label="Upload Sample (5-15 seconds)", type="filepath"),
        gr.Audio(label="Upload Full Song", type="filepath")
    ],
    outputs=[
        gr.JSON(label="Analysis Results"),
        gr.Audio(label="Sample Playback"),
        gr.Audio(label="Song Playback"),
        gr.Image(label="Sample Detection Timeline")
    ],
    title="Sample Detection Demo",
    description="Upload a sample and a song to find where the sample appears in the song"
)

# For Lightning AI
app = demo.app

if __name__ == "__main__":
    logger.info("Starting application")
    demo.launch()
