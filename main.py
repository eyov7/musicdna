import gradio as gr
import numpy as np
import librosa
import logging
from core.analyzers import GranularSampleDetector
import torch
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDNAApp:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = GranularSampleDetector(device=device)
        logger.info(f"Initialized MusicDNA App. Using device: {device}")

    def format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS.ms format"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"

    def process_audio(self, sample_path: str, song_path: str) -> str:
        """Process audio files and return analysis results"""
        try:
            # Load audio files
            logger.info("Loading audio files...")
            sample_y, sr = librosa.load(sample_path, sr=22050)
            song_y, _ = librosa.load(song_path, sr=22050)

            # Analyze sample
            logger.info("Analyzing sample...")
            sample_analysis = self.detector.analyze(sample_y)

            # Find matches in song
            logger.info("Searching for matches...")
            matches = self.detector.find_in_track(sample_analysis, song_y, sr)

            # Format results
            output = "🎵 MusicDNA Analysis Results 🎵\n\n"
            
            # Sample Analysis
            output += "📊 Sample Analysis:\n"
            output += f"Duration: {self.format_time(sample_analysis['metadata']['duration'])}\n"
            output += f"Sample Rate: {sample_analysis['metadata']['sample_rate']} Hz\n\n"
            
            # Stem Analysis
            output += "🎼 Stem Analysis:\n"
            for stem_name, stem_data in sample_analysis['stems'].items():
                output += f"\n🎵 {stem_name.upper()} Stem:\n"
                features = stem_data['features']
                output += f"- RMS Energy: {features['rms']:.3f}\n"
                output += f"- Peak Amplitude: {features['peak']:.3f}\n"
                output += f"- Zero Crossings: {features['zero_crossings']}\n"
            
            # Match Results
            if matches:
                output += f"\n🎯 Found {len(matches)} potential matches:\n"
                for i, match in enumerate(matches, 1):
                    output += f"\n📍 Match {i}:\n"
                    output += f"Time: {self.format_time(match['time_start'])} to {self.format_time(match['time_end'])}\n"
                    output += f"Overall Confidence: {match['total_confidence']:.1%}\n"
                    
                    # Confidence per stem
                    output += "\nConfidence by stem:\n"
                    for stem, conf in match['confidence'].items():
                        output += f"- {stem}: {conf['total']:.1%}\n"
                        output += f"  • Spectral: {conf['spectral']:.1%}\n"
                        output += f"  • MIDI: {conf['midi']:.1%}\n"
                        output += f"  • Features: {conf['features']:.1%}\n"
                    
                    # Transformations
                    if any(match['transformations'].values()):
                        output += "\nDetected Transformations:\n"
                        for stem, trans in match['transformations'].items():
                            if trans:
                                output += f"- {stem}:\n"
                                if 'pitch_shift' in trans:
                                    output += f"  • Pitch shift: {trans['pitch_shift']:.1f} semitones\n"
                                if 'time_stretch' in trans:
                                    output += f"  • Time stretch: {trans['time_stretch']:.1%}\n"
            else:
                output += "\n❌ No significant matches found."

            return output

        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            logger.error(f"Traceback:", exc_info=True)
            return f"❌ Error processing audio: {str(e)}"

    def create_interface(self):
        """Create Gradio interface"""
        interface = gr.Interface(
            fn=self.process_audio,
            inputs=[
                gr.Audio(label="Upload Sample (5-15 seconds)", type="filepath"),
                gr.Audio(label="Upload Full Song", type="filepath")
            ],
            outputs=gr.Textbox(label="Analysis Results", lines=25),
            title="🧬 MusicDNA - Advanced Sample Detection",
            description="""
            ## 🎵 MusicDNA Sample Detection System

            This system performs granular multi-level analysis to find samples and their transformations:

            ### 🔍 Analysis Levels:

            1. 📊 Full Audio Analysis
               - Spectral features
               - Energy distribution
               - Overall characteristics

            2. 🎼 Stem Separation
               - 🥁 Drums
               - 🎸 Bass
               - 🎤 Vocals
               - 🎹 Other instruments

            3. 🎵 Per-Stem Analysis
               - Spectral matching
               - MIDI pattern detection
               - Feature comparison

            4. 🔄 Transformation Detection
               - Pitch shifting
               - Time stretching
               - Audio modifications

            Upload a sample and a song to analyze their musical DNA and find potential matches!
            """,
            theme=gr.themes.Soft(),
            css=".gradio-container {max-width: 800px; margin: auto}"
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
