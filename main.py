import gradio as gr
import numpy as np
import librosa
import logging
from core.analyzers import SpectralAnalyzer, StemAnalyzer
import torch
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDNAApp:
    def __init__(self):
        # Initialize analyzers
        self.spectral_analyzer = SpectralAnalyzer()
        self.stem_analyzer = StemAnalyzer(device="cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized analyzers. Using device: {self.stem_analyzer.device}")

    def process_audio(self, sample_path: str, song_path: str) -> str:
        """
        Process audio files and return analysis results
        """
        try:
            # Load audio files
            logger.info("Loading audio files...")
            sample_y, sr = librosa.load(sample_path, sr=22050)
            song_y, _ = librosa.load(song_path, sr=22050)

            # Analyze sample
            logger.info("Analyzing sample...")
            sample_spectral = self.spectral_analyzer.analyze(sample_y)
            sample_stems = self.stem_analyzer.analyze(sample_y)

            # Analyze song (just a portion for initial comparison)
            song_duration = len(song_y) / sr
            logger.info(f"Song duration: {song_duration:.2f} seconds")
            
            # Take first minute for initial analysis
            analysis_duration = min(60, song_duration)
            song_samples = int(analysis_duration * sr)
            song_excerpt = song_y[:song_samples]
            
            logger.info("Analyzing song excerpt...")
            song_spectral = self.spectral_analyzer.analyze(song_excerpt)
            song_stems = self.stem_analyzer.analyze(song_excerpt)

            # Format results
            output = "Analysis Results:\n\n"
            
            # Sample Analysis
            output += "Sample Analysis:\n"
            output += f"Duration: {len(sample_y)/sr:.2f} seconds\n"
            output += "Features extracted:\n"
            for feature in sample_spectral:
                if feature != 'metadata':
                    output += f"- {feature}\n"
            
            output += "\nStem Analysis:\n"
            for stem_name in sample_stems['stem_features']:
                output += f"- {stem_name} stem extracted and analyzed\n"
            
            # Song Analysis
            output += f"\nSong Analysis (first {analysis_duration:.1f} seconds):\n"
            output += f"Total Duration: {song_duration:.2f} seconds\n"
            output += "Features extracted:\n"
            for feature in song_spectral:
                if feature != 'metadata':
                    output += f"- {feature}\n"
                    
            output += "\nStem Analysis:\n"
            for stem_name in song_stems['stem_features']:
                output += f"- {stem_name} stem extracted and analyzed\n"

            return output

        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error processing audio: {str(e)}"

    def create_interface(self):
        """Create Gradio interface"""
        interface = gr.Interface(
            fn=self.process_audio,
            inputs=[
                gr.Audio(label="Upload Sample (5-15 seconds)", type="filepath"),
                gr.Audio(label="Upload Full Song", type="filepath")
            ],
            outputs=gr.Textbox(label="Analysis Results", lines=20),
            title="MusicDNA - Advanced Sample Detection",
            description="""
            ## MusicDNA Sample Detection System
            
            This system performs multi-level analysis of audio samples:
            - Spectral Analysis (mel-spectrograms, MFCCs, chroma features)
            - Stem Separation (drums, bass, vocals, other)
            - Feature Extraction per stem
            
            Upload a sample and a song to analyze their musical DNA.
            """
        )
        return interface

def main():
    # Create app instance
    app = MusicDNAApp()
    
    # Create interface
    demo = app.create_interface()
    
    # Get port from environment (for Lightning AI) or use default
    port = int(os.environ.get('PORT', 7860))
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Required for Lightning AI
        server_port=port,
        share=True
    )

if __name__ == "__main__":
    main()
