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
            ## 🎵 Upload a sample and a full song to detect where the sample appears
            
            This tool uses advanced audio analysis to:
            - 🎼 Detect samples across different musical elements
            - 🔍 Identify transformations (pitch shifts, time stretches)
            - 📊 Provide confidence scores for matches
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
            results = ["🎯 Found potential matches:\n"]
            
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
    app.interface.launch(
        share=False,
        enable_queue=False
    )

if __name__ == "__main__":
    main()
