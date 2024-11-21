import gradio as gr
import numpy as np
import logging
import matplotlib.pyplot as plt
from core.analyzers.granular_detector import GranularSampleDetector
from core.analyzers.stem_analyzer import StemAnalyzer
from pathlib import Path
import librosa
import librosa.display
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDNAApp:
    def __init__(self):
        """Initialize the MusicDNA application with enhanced analyzers."""
        self.detector = GranularSampleDetector(stem_weights={
            'drums': 0.3,
            'bass': 0.3,
            'vocals': 0.2,
            'other': 0.2
        })
        self.stem_analyzer = StemAnalyzer()
        self.setup_interface()

    def create_visualization(self, audio_data, sample_rate):
        """Create spectrogram visualization."""
        fig, ax = plt.subplots(figsize=(10, 4))
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        return fig

    def analyze_audio(self, audio_data, sample_rate):
        """Perform comprehensive audio analysis."""
        try:
            # Create fingerprint
            analysis = self.stem_analyzer.analyze(audio_data)
            
            # Get primary stem fingerprint
            primary_stem = analysis['primary_stem']
            primary_stem_data = analysis['stems'][primary_stem]
            fingerprint = primary_stem_data['fingerprint']
            
            # Extract basic audio features
            features = {
                'tempo': librosa.beat.tempo(y=audio_data, sr=sample_rate)[0],
                'rms_energy': np.mean(librosa.feature.rms(y=audio_data)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            }
            
            return {
                'fingerprint': fingerprint,
                'features': {
                    'tempo': f"🎵 Tempo: {features['tempo']:.1f} BPM",
                    'energy': f"💪 Energy: {features['rms_energy']:.3f}",
                    'spectral_centroid': f"🎼 Brightness: {features['spectral_centroid']:.1f} Hz",
                    'primary_stem': f"🎸 Primary Stem: {primary_stem}"
                }
            }
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            return None

    def process_audio(self, sample_audio, song_audio):
        """Process audio files and return detailed analysis."""
        try:
            if sample_audio is None or song_audio is None:
                return "Please provide both sample and song audio files."

            sample_data, sample_rate = sample_audio
            song_data, song_rate = song_audio

            # Analyze sample
            sample_analysis = self.analyze_audio(sample_data, sample_rate)
            if not sample_analysis:
                return "Error analyzing sample audio."

            # Analyze song
            song_analysis = self.analyze_audio(song_data, song_rate)
            if not song_analysis:
                return "Error analyzing song audio."

            # Find sample matches
            matches = self.stem_analyzer.find_sample(
                sample_data,
                self.stem_analyzer.separate_stems(song_data)
            )

            # Format results
            results = []
            results.append("🧬 MusicDNA Analysis Results\n")
            results.append("\n📊 Sample Analysis:")
            for key, value in sample_analysis['features'].items():
                results.append(f"  {value}")

            results.append("\n🎵 Song Analysis:")
            for key, value in song_analysis['features'].items():
                results.append(f"  {value}")

            results.append("\n🎯 Sample Matches Found:")
            if matches:
                for match in matches:
                    results.append(
                        f"  Match at {match['start']/sample_rate:.2f}s "
                        f"in {match['stem']} stem "
                        f"(confidence: {match['confidence']:.2%})"
                    )
            else:
                results.append("  No significant matches found")

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
            return f"Error processing audio: {str(e)}"

    def setup_interface(self):
        """Setup the Gradio interface."""
        self.interface = gr.Interface(
            fn=self.process_audio,
            inputs=[
                gr.Audio(label="Sample Audio", type="numpy"),
                gr.Audio(label="Song Audio", type="numpy")
            ],
            outputs=gr.Textbox(label="Analysis Results"),
            title="MusicDNA - Advanced Sample Detection",
            description="""
            Upload a sample audio file and a song to analyze potential sample usage.
            The system will perform granular analysis using stem separation and multi-level fingerprinting.
            """
        )

def main():
    app = MusicDNAApp()
    app.interface.launch(debug=True)

if __name__ == "__main__":
    main()
