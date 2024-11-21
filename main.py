import gradio as gr
import logging
import librosa
import numpy as np
import soundfile as sf
import os
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

def process_audio(sample_path: str, song_path: str) -> dict:
    """Process audio files and return matches with visualizations"""
    try:
        # Load and process audio files
        sample_y, sr = librosa.load(sample_path)
        song_y, _ = librosa.load(song_path, sr=sr)
        
        # Get stems for both
        sample_stems, _ = demucs.separate_stems(sample_path)
        song_stems, _ = demucs.separate_stems(song_path)
        
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
        
        # Format output
        output_text = "Found matches:\n"
        for match in matches:
            output_text += f"Time: {match.start_time:.2f}s, Duration: {match.end_time - match.start_time:.2f}s, Confidence: {match.confidence:.2%}\n"
        
        # Create interactive visualizations
        fig_freq = go.Figure(data=[go.Scatter(x=np.linspace(0, song_length, len(similarity_scores)), y=similarity_scores)])
        fig_freq.update_layout(title='Frequency Analysis', xaxis_title='Time (seconds)', yaxis_title='Similarity Score')
        
        fig_pattern = go.Figure(data=[go.Scatter(x=np.linspace(0, song_length, len(similarity_scores)), y=similarity_scores)])
        fig_pattern.update_layout(title='Pattern Analysis', xaxis_title='Time (seconds)', yaxis_title='Similarity Score')
        
        fig_mix = go.Figure(data=[go.Scatter(x=np.linspace(0, song_length, len(similarity_scores)), y=similarity_scores)])
        fig_mix.update_layout(title='Mix Density Analysis', xaxis_title='Time (seconds)', yaxis_title='Similarity Score')
        
        return {
            "output": output_text,
            "freq_viz": fig_freq,
            "pattern_viz": fig_pattern,
            "mix_viz": fig_mix
        }
        
    except Exception as e:
        logger.error(f"Error in sample analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks() as interface:
        gr.Markdown("# MusicDNA - Advanced Sample Detection")
        
        with gr.Row():
            with gr.Column():
                sample_input = gr.Audio(label="Upload Sample")
                song_input = gr.Audio(label="Upload Full Song")
                analyze_btn = gr.Button("Analyze")
            
            with gr.Column():
                output_text = gr.Textbox(label="Detection Results")
                
        with gr.Row():
            freq_plot = gr.Plot(label="Frequency Analysis")
            pattern_plot = gr.Plot(label="Pattern Analysis")
            
        with gr.Row():
            mix_plot = gr.Plot(label="Mix Density Analysis")
        
        analyze_btn.click(
            fn=process_audio,
            inputs=[sample_input, song_input],
            outputs=[output_text, freq_plot, pattern_plot, mix_plot]
        )
        
    return interface

# Create Gradio interface
demo = create_interface()

# For Lightning AI
app = demo.app

if __name__ == "__main__":
    logger.info("Starting application")
    demo.launch()
