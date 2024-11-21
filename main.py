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
            fingerprint = self.stem_analyzer.create_sample_fingerprint(audio_data)
            
            # Extract features for display
            features = fingerprint['audio_features']
            
            return {
                'fingerprint': fingerprint,
                'features': {
                    'tempo': f"🎵 Tempo: {features.get('tempo', 0):.1f} BPM",
                    'energy': f"💪 Energy: {features.get('rms_energy', 0):.3f}",
                    'spectral_centroid': f"🎼 Brightness: {features.get('spectral_centroid', 0):.1f} Hz"
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

            # Compare fingerprints
            confidence_scores = self.stem_analyzer.compare_fingerprints(
                sample_analysis['fingerprint'],
                song_analysis['fingerprint']
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

            results.append("\n🎯 Match Confidence Scores:")
            results.append(f"  Overall: {confidence_scores.get('overall', 0):.2%}")
            
            if 'stems' in confidence_scores:
                results.append("\n🎼 Per-Stem Confidence:")
                stem_emojis = {
                    'drums': '🥁',
                    'bass': '🎸',
                    'vocals': '🎤',
                    'other': '🎹'
                }
                for stem, score in confidence_scores['stems'].items():
                    emoji = stem_emojis.get(stem, '🎵')
                    results.append(f"  {emoji} {stem.title()}: {score:.2%}")

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return f"Error processing audio: {str(e)}"

    def setup_interface(self):
        """Set up the enhanced Gradio interface."""
        self.interface = gr.Interface(
            fn=self.process_audio,
            inputs=[
                gr.Audio(label="Sample Audio", type="numpy"),
                gr.Audio(label="Full Song", type="numpy")
            ],
            outputs=[
                gr.Textbox(label="Analysis Results")
            ],
            title="🧬 MusicDNA - Advanced Sample Detection",
            description="""
            ## 🎵 MusicDNA: Advanced Audio DNA Analysis System

            This system performs comprehensive audio analysis using state-of-the-art techniques:

            ### 🔬 Analysis Capabilities:

            1. 📊 Spectral Analysis
               - Multiple spectrogram representations
               - Fourier component analysis
               - Frequency distribution patterns

            2. 🎼 Stem Separation & Analysis
               - 🥁 Drums: Rhythm patterns & timing
               - 🎸 Bass: Harmonic foundation
               - 🎤 Vocals: Melodic elements
               - 🎹 Other: Additional components

            3. 🎵 Feature Extraction
               - Tempo detection
               - Energy analysis
               - Spectral characteristics
               - MIDI pattern recognition

            4. 🧬 DNA Matching
               - Multi-level fingerprint comparison
               - Per-stem confidence scoring
               - Transformation detection
               - Pattern recognition

            ### 💡 How to Use:

            1. **Upload Sample**: Add the audio sample you want to find
            2. **Upload Song**: Add the full song to analyze
            3. **View Results**: Get detailed analysis including:
               - Audio characteristics
               - Match confidence scores
               - Per-stem analysis
               - Transformation details

            ### 🎯 Best Practices:
            - Use high-quality audio files
            - Trim samples to relevant sections
            - Allow processing time for detailed analysis

            Start analyzing your music's DNA! 🎵
            """,
            theme=gr.themes.Soft(),
            allow_flagging="never"
        )

def main():
    app = MusicDNAApp()
    app.interface.launch()

if __name__ == "__main__":
    main()
