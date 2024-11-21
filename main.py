import gradio as gr
import numpy as np
import logging
from core.analyzers.granular_detector import GranularSampleDetector
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDNAApp:
    def __init__(self):
        """Initialize the MusicDNA application."""
        # Initialize detector with default stem weights
        self.detector = GranularSampleDetector(stem_weights={
            'drums': 0.3,
            'bass': 0.3,
            'vocals': 0.2,
            'other': 0.2
        })
        self.setup_interface()

    def setup_interface(self):
        """Set up the Gradio interface."""
        self.interface = gr.Interface(
            fn=self.process_audio,
            inputs=[
                gr.Audio(label="Sample Audio", type="numpy"),
                gr.Audio(label="Full Song", type="numpy")
            ],
            outputs=gr.Textbox(label="Analysis Results"),
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

            ### 💡 How it Works:

            1. **Sample Analysis**: Your sample is analyzed across multiple dimensions:
               - Spectral characteristics
               - Stem separation
               - MIDI patterns
               - Audio features

            2. **Track Analysis**: The full song is processed similarly

            3. **Pattern Matching**: Advanced algorithms compare the sample and track:
               - Multi-level feature matching
               - Transformation detection
               - Confidence scoring

            4. **Results**: You get detailed insights about:
               - Where samples appear
               - How they've been modified
               - Confidence levels
               - Stem-specific matches

            Upload a sample and a song to analyze their musical DNA! 🎵
            """,
            theme=gr.themes.Soft(),
            allow_flagging="never"
        )

    def format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS.ms format."""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"

    def format_transformations(self, transformations: list) -> str:
        """Format transformation information with emojis."""
        if not transformations:
            return "🎯 No transformations detected"
        
        emoji_map = {
            'pitch_shift': '🎹',
            'time_stretch': '⏱️'
        }
        
        return " ".join([f"{emoji_map.get(t, '❓')} {t.replace('_', ' ').title()}" 
                        for t in transformations])

    def process_audio(self, sample_audio, song_audio):
        """Process audio files and detect samples."""
        try:
            if sample_audio is None or song_audio is None:
                return "⚠️ Please provide both a sample and a song audio file."

            sample_y, sr = sample_audio
            song_y, _ = song_audio

            # Analyze sample
            sample_analysis = self.detector.analyze(sample_y)
            
            # Find matches
            matches = self.detector.find_in_track(sample_analysis, song_y, sr)

            if not matches:
                return "❌ No matches found in the track."

            # Format results
            results = ["🎵 MusicDNA Analysis Results 🎵\n"]
            
            # Sample Analysis Summary
            results.append("\n📊 Sample Analysis:")
            results.append(f"Duration: {self.format_time(sample_analysis['metadata']['duration'])}")
            results.append(f"Sample Rate: {sample_analysis['metadata']['sample_rate']} Hz\n")
            
            # Match Results
            results.append("🎯 Found potential matches:\n")
            
            for i, match in enumerate(matches, 1):
                confidence = match.get('confidence', 0) * 100
                stem = match.get('stem', 'full track')
                transformations = match.get('transformations', [])
                
                match_info = [
                    f"\n📍 Match {i}:",
                    f"🎯 Location: {stem}",
                    f"✨ Confidence: {confidence:.1f}%",
                    f"🔄 {self.format_transformations(transformations)}"
                ]
                
                results.extend(match_info)

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return f"⚠️ An error occurred during processing: {str(e)}"

def main():
    app = MusicDNAApp()
    app.interface.queue()  # Use queue instead of enable_queue
    app.interface.launch(
        share=False
    )

if __name__ == "__main__":
    main()
