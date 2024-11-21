import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Optional, Tuple
import io
import base64

logger = logging.getLogger(__name__)

class AudioVisualizer:
    def __init__(self,
                 sr: int = 44100,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Initialize audio visualizer
        
        Args:
            sr: Sample rate
            hop_length: Number of samples between frames
            n_mels: Number of mel bands to generate
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Try to import plotly, fallback to matplotlib if not available
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            self.use_plotly = True
            self.go = go
            self.make_subplots = make_subplots
        except ImportError:
            self.use_plotly = False
            logger.warning("Plotly not available, falling back to matplotlib")

    def plot_to_image(self, fig):
        """Convert matplotlib figure to base64 image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf

    def plot_frequency_bands(self,
                           sample: np.ndarray,
                           song: np.ndarray,
                           freq_weights: np.ndarray,
                           matches: List[Dict],
                           title: str = "Frequency Band Analysis"):
        """Create frequency band visualization"""
        try:
            # Calculate spectrograms
            sample_spec = librosa.feature.melspectrogram(
                y=sample,
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            song_spec = librosa.feature.melspectrogram(
                y=song,
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # Convert to dB scale
            sample_db = librosa.power_to_db(sample_spec, ref=np.max)
            song_db = librosa.power_to_db(song_spec, ref=np.max)
            
            if self.use_plotly:
                # Create plotly figure
                fig = self.make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(
                        "Sample Spectrogram",
                        "Song Spectrogram",
                        "Frequency Band Weights"
                    ),
                    vertical_spacing=0.1
                )
                
                # Add spectrograms and weights
                fig.add_trace(
                    self.go.Heatmap(z=sample_db, colorscale='Viridis'),
                    row=1, col=1
                )
                fig.add_trace(
                    self.go.Heatmap(z=song_db, colorscale='Viridis'),
                    row=2, col=1
                )
                fig.add_trace(
                    self.go.Bar(x=np.arange(len(freq_weights)), y=freq_weights),
                    row=3, col=1
                )
                
                # Update layout
                fig.update_layout(height=800, title=title)
                return fig
                
            else:
                # Create matplotlib figure
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
                fig.suptitle(title)
                
                # Plot spectrograms
                img1 = ax1.imshow(sample_db, aspect='auto', origin='lower')
                ax1.set_title("Sample Spectrogram")
                plt.colorbar(img1, ax=ax1)
                
                img2 = ax2.imshow(song_db, aspect='auto', origin='lower')
                ax2.set_title("Song Spectrogram")
                plt.colorbar(img2, ax=ax2)
                
                # Plot frequency weights
                ax3.bar(np.arange(len(freq_weights)), freq_weights)
                ax3.set_title("Frequency Band Weights")
                
                # Add match markers
                for match in matches:
                    start_frame = int(match['start_time'] * self.sr / self.hop_length)
                    duration = match.get('duration', 1.0)
                    end_frame = int((match['start_time'] + duration) * self.sr / self.hop_length)
                    ax2.axvspan(start_frame, end_frame, color='red', alpha=0.2)
                
                plt.tight_layout()
                return self.plot_to_image(fig)
                
        except Exception as e:
            logger.error(f"Error creating frequency visualization: {str(e)}")
            raise

    def plot_pattern_map(self,
                        song: np.ndarray,
                        patterns: List[Dict],
                        matches: List[Dict],
                        title: str = "Pattern Analysis"):
        """Create pattern map visualization"""
        try:
            # Calculate self-similarity matrix
            S = librosa.feature.melspectrogram(
                y=song,
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            S = librosa.power_to_db(S, ref=np.max)
            
            # Create pattern timeline
            timeline = np.zeros(S.shape[1])
            for pattern in patterns:
                for occurrence in pattern['patterns']:
                    start_frame = int(occurrence['time1'] * self.sr / self.hop_length)
                    duration_frames = int(occurrence['duration'] * self.sr / self.hop_length)
                    timeline[start_frame:start_frame + duration_frames] += 1
            
            if self.use_plotly:
                fig = self.make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Self-Similarity Matrix", "Pattern Timeline"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    self.go.Heatmap(z=S, colorscale='Viridis'),
                    row=1, col=1
                )
                fig.add_trace(
                    self.go.Scatter(y=timeline, mode='lines'),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title=title)
                return fig
                
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle(title)
                
                img = ax1.imshow(S, aspect='auto', origin='lower')
                ax1.set_title("Self-Similarity Matrix")
                plt.colorbar(img, ax=ax1)
                
                ax2.plot(timeline)
                ax2.set_title("Pattern Timeline")
                
                # Add match markers
                for match in matches:
                    start_frame = int(match['start_time'] * self.sr / self.hop_length)
                    duration = match.get('duration', 1.0)
                    end_frame = int((match['start_time'] + duration) * self.sr / self.hop_length)
                    ax2.axvspan(start_frame, end_frame, color='red', alpha=0.2)
                
                plt.tight_layout()
                return self.plot_to_image(fig)
                
        except Exception as e:
            logger.error(f"Error creating pattern visualization: {str(e)}")
            raise

    def plot_mix_density(self,
                        density: np.ndarray,
                        matches: List[Dict],
                        confidence_scores: np.ndarray,
                        title: str = "Mix Density Analysis"):
        """Create mix density visualization"""
        try:
            if self.use_plotly:
                fig = self.make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Mix Density", "Confidence Scores"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    self.go.Scatter(y=density, mode='lines', name='Mix Density'),
                    row=1, col=1
                )
                fig.add_trace(
                    self.go.Scatter(y=confidence_scores, mode='lines', name='Confidence'),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title=title)
                return fig
                
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                fig.suptitle(title)
                
                ax1.plot(density)
                ax1.set_title("Mix Density")
                
                ax2.plot(confidence_scores)
                ax2.set_title("Confidence Scores")
                
                # Add match markers
                for match in matches:
                    start_frame = int(match['start_time'] * self.sr / self.hop_length)
                    duration = match.get('duration', 1.0)
                    end_frame = int((match['start_time'] + duration) * self.sr / self.hop_length)
                    for ax in [ax1, ax2]:
                        ax.axvspan(start_frame, end_frame, color='red', alpha=0.2)
                
                plt.tight_layout()
                return self.plot_to_image(fig)
                
        except Exception as e:
            logger.error(f"Error creating density visualization: {str(e)}")
            raise
