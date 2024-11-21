import gradio as gr
import numpy as np
import librosa
import logging
from core.analyzers import SpectralAnalyzer, StemAnalyzer, MIDIAnalyzer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDNAApp:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize analyzers
        self.spectral_analyzer = SpectralAnalyzer()
        self.stem_analyzer = StemAnalyzer(device=device)
        self.midi_analyzer = MIDIAnalyzer(device=device)
        logger.info(f"Initialized analyzers. Using device: {device}")

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
            sample_midi = self.midi_analyzer.analyze(sample_y, sr)

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
            song_midi = self.midi_analyzer.analyze(song_excerpt, sr)

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
                
            output += "\nMIDI Analysis:\n"
            output += f"- Total notes detected: {sample_midi['metadata']['num_notes']}\n"
            if sample_midi['metadata'].get('pitch_range'):
                output += f"- Pitch range: {sample_midi['metadata']['pitch_range']['min']} to {sample_midi['metadata']['pitch_range']['max']}\n"
            
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
                
            output += "\nMIDI Analysis:\n"
            output += f"- Total notes detected: {song_midi['metadata']['num_notes']}\n"
            if song_midi['metadata'].get('pitch_range'):
                output += f"- Pitch range: {song_midi['metadata']['pitch_range']['min']} to {song_midi['metadata']['pitch_range']['max']}\n"

            # Compare MIDI data
            midi_similarity = self.midi_analyzer.compare_midi(sample_midi, song_midi)
            output += f"\nMIDI Similarity Score: {midi_similarity:.2%}\n"

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
            - MIDI Analysis (note events, pitch, timing)
            - Feature Extraction per stem
            
            Upload a sample and a song to analyze their musical DNA.
            """
        )
        return interface

def main():
    # Create app instance
    app = MusicDNAApp()
    
    # Create and launch interface
    demo = app.create_interface()
    demo.launch(server_port=7860)

if __name__ == "__main__":
    main()
