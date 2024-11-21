import gradio as gr
import numpy as np
import librosa
import logging
from core.analyzers import GranularSampleDetector
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDNAApp:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = GranularSampleDetector(device=device)
        logger.info(f"Initialized MusicDNA App. Using device: {device}")

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
            sample_analysis = self.detector.analyze_sample(sample_y, sr)

            # Find matches in song
            logger.info("Searching for matches...")
            matches = self.detector.find_in_track(sample_analysis, song_y, sr)

            # Format results
            output = "Analysis Results:\n\n"
            
            # Sample Analysis
            output += "Sample Analysis:\n"
            output += f"Duration: {sample_analysis['metadata']['duration']:.2f} seconds\n"
            
            # Spectral Analysis
            output += "\nSpectral Features:\n"
            for feature in sample_analysis['full_spectral']:
                if feature != 'metadata':
                    output += f"- {feature}\n"
            
            # Stem Analysis
            output += "\nStem Analysis:\n"
            for stem_name in sample_analysis['stems']['stem_features']:
                output += f"- {stem_name} stem extracted and analyzed\n"
                midi_data = sample_analysis['midi_per_stem'][stem_name]
                output += f"  * Notes detected: {midi_data['metadata']['num_notes']}\n"
                if midi_data['metadata'].get('pitch_range'):
                    output += f"  * Pitch range: {midi_data['metadata']['pitch_range']['min']} to {midi_data['metadata']['pitch_range']['max']}\n"
            
            # Match Results
            if matches:
                output += f"\nFound {len(matches)} potential matches:\n"
                for i, match in enumerate(matches, 1):
                    output += f"\nMatch {i}:\n"
                    output += f"Time: {match['time_start']:.2f}s to {match['time_end']:.2f}s\n"
                    output += f"Overall Confidence: {match['overall_confidence']:.2%}\n"
                    
                    output += "Confidence per stem:\n"
                    for stem, conf in match['confidence_per_stem'].items():
                        output += f"- {stem}: {conf:.2%}\n"
                        
                    output += "MIDI matches per stem:\n"
                    for stem, conf in match['midi_matches'].items():
                        output += f"- {stem}: {conf:.2%}\n"
            else:
                output += "\nNo significant matches found."

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
            
            This system performs granular multi-level analysis:
            1. Full Spectral Analysis
               - Mel-spectrograms
               - MFCCs
               - Chroma features
               
            2. Stem Separation
               - Drums
               - Bass
               - Vocals
               - Other instruments
               
            3. MIDI Analysis per stem
               - Note events
               - Pitch tracking
               - Timing analysis
            
            Upload a sample and a song to analyze their musical DNA and find potential matches.
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
