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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize processors
demucs = DemucsProcessor()
basic_pitch = BasicPitchConverter()

def create_similarity_plot(similarity_matrix, window_size):
    """Create visualization of similarity over time"""
    plt.figure(figsize=(12, 4))
    plt.plot(np.max(similarity_matrix, axis=0))
    plt.title('Sample Similarity Over Time')
    plt.xlabel('Time (frames)')
    plt.ylabel('Similarity Score')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold')
    plt.legend()
    
    with NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        plt.savefig(temp_file.name)
        plt.close()
        return temp_file.name

def find_sample_timestamps(similarity_matrix, threshold=0.8):
    """Find timestamps where sample appears in the song"""
    peak_scores = np.max(similarity_matrix, axis=0)
    matches = np.where(peak_scores > threshold)[0]
    
    # Group consecutive frames
    if len(matches) == 0:
        return []
    
    timestamps = []
    current_start = matches[0]
    
    for i in range(1, len(matches)):
        if matches[i] - matches[i-1] > 1:
            timestamps.append({
                'start_frame': current_start,
                'end_frame': matches[i-1],
                'confidence': float(np.mean(peak_scores[current_start:matches[i-1]]))
            })
            current_start = matches[i]
    
    # Add last group
    timestamps.append({
        'start_frame': current_start,
        'end_frame': matches[-1],
        'confidence': float(np.mean(peak_scores[current_start:matches[-1]]))
    })
    
    return timestamps

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
        sample_chroma = librosa.feature.chroma_cqt(y=sample_stems['other'].squeeze(), sr=sr)
        song_chroma = librosa.feature.chroma_cqt(y=song_stems['other'].squeeze(), sr=sr)
        
        # Compute similarity matrix
        similarity_matrix = librosa.segment.cross_similarity(
            sample_chroma, 
            song_chroma,
            mode='affinity'
        )
        
        # Find timestamps of potential matches
        matches = find_sample_timestamps(similarity_matrix)
        
        # Convert frames to timestamps
        hop_length = 512  # Default hop length in librosa
        frame_time = hop_length / sr
        
        for match in matches:
            match['start_time'] = match['start_frame'] * frame_time
            match['end_time'] = match['end_frame'] * frame_time
            del match['start_frame']
            del match['end_frame']
        
        # Create visualizations
        similarity_plot = create_similarity_plot(similarity_matrix, len(sample_chroma[0]))
        
        # Prepare results
        results = {
            "Sample Length": f"{librosa.get_duration(y=sample_y, sr=sr):.2f} seconds",
            "Song Length": f"{librosa.get_duration(y=song_y, sr=sr):.2f} seconds",
            "Potential Matches": matches,
            "Number of Matches": len(matches)
        }
        
        logger.info("Analysis completed successfully")
        logger.info(f"Results: {results}")
        
        return (
            results,
            sample_audio,
            full_song,
            similarity_plot
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
