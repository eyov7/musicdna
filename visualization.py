import numpy as np
import librosa
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    def plot_frequency_bands(self,
                           sample: np.ndarray,
                           song: np.ndarray,
                           freq_weights: np.ndarray,
                           matches: List[Dict],
                           title: str = "Frequency Band Analysis") -> go.Figure:
        """
        Create interactive frequency band visualization
        
        Args:
            sample: Sample audio array
            song: Full song audio array
            freq_weights: Frequency band weights
            matches: List of detected matches
            title: Plot title
            
        Returns:
            Plotly figure object
        """
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
            
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    "Sample Spectrogram",
                    "Song Spectrogram",
                    "Frequency Band Weights"
                ),
                vertical_spacing=0.1
            )
            
            # Plot sample spectrogram
            fig.add_trace(
                go.Heatmap(
                    z=sample_db,
                    colorscale='Viridis',
                    showscale=False,
                    name="Sample"
                ),
                row=1, col=1
            )
            
            # Plot song spectrogram
            fig.add_trace(
                go.Heatmap(
                    z=song_db,
                    colorscale='Viridis',
                    showscale=True,
                    name="Song"
                ),
                row=2, col=1
            )
            
            # Plot frequency weights
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(freq_weights)),
                    y=freq_weights,
                    name="Weights"
                ),
                row=3, col=1
            )
            
            # Add match markers
            for match in matches:
                start_time = match['start_time']
                duration = match.get('duration', 1.0)
                confidence = match['confidence']
                
                # Convert time to frame index
                start_frame = int(start_time * self.sr / self.hop_length)
                end_frame = int((start_time + duration) * self.sr / self.hop_length)
                
                # Add rectangle shapes to mark matches
                fig.add_shape(
                    type="rect",
                    x0=start_frame,
                    x1=end_frame,
                    y0=0,
                    y1=self.n_mels,
                    line=dict(
                        color="rgba(255, 0, 0, 0.5)",
                        width=2,
                    ),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                showlegend=True,
                xaxis3_title="Frequency Band",
                yaxis3_title="Weight",
                xaxis2_title="Time (frames)",
                yaxis2_title="Mel Band",
                xaxis1_title="Time (frames)",
                yaxis1_title="Mel Band"
            )
            
            # Add hover information
            fig.update_traces(
                hovertemplate="Time: %{x}<br>Frequency: %{y}<br>Magnitude: %{z}<extra></extra>"
            )
            
            logger.info("Created frequency band visualization")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating frequency visualization: {str(e)}")
            raise

    def plot_pattern_map(self,
                        song: np.ndarray,
                        patterns: List[Dict],
                        matches: List[Dict],
                        title: str = "Pattern Analysis") -> go.Figure:
        """
        Create interactive pattern map visualization
        
        Args:
            song: Full song audio array
            patterns: List of detected patterns
            matches: List of detected matches
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate self-similarity matrix
            S = librosa.feature.melspectrogram(
                y=song,
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            S = librosa.power_to_db(S, ref=np.max)
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Self-Similarity Matrix",
                    "Pattern Timeline"
                ),
                vertical_spacing=0.1
            )
            
            # Plot self-similarity matrix
            fig.add_trace(
                go.Heatmap(
                    z=S,
                    colorscale='Viridis',
                    showscale=True,
                    name="Similarity"
                ),
                row=1, col=1
            )
            
            # Create pattern timeline
            timeline = np.zeros(S.shape[1])
            for pattern in patterns:
                for occurrence in pattern['patterns']:
                    start_frame = int(occurrence['time1'] * self.sr / self.hop_length)
                    duration_frames = int(occurrence['duration'] * self.sr / self.hop_length)
                    timeline[start_frame:start_frame + duration_frames] += 1
            
            # Plot pattern timeline
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(timeline)),
                    y=timeline,
                    name="Patterns",
                    mode='lines'
                ),
                row=2, col=1
            )
            
            # Add match markers
            for match in matches:
                start_time = match['start_time']
                duration = match.get('duration', 1.0)
                
                # Convert time to frame index
                start_frame = int(start_time * self.sr / self.hop_length)
                end_frame = int((start_time + duration) * self.sr / self.hop_length)
                
                # Add rectangle shapes to mark matches
                fig.add_shape(
                    type="rect",
                    x0=start_frame,
                    x1=end_frame,
                    y0=0,
                    y1=max(timeline) * 1.1,
                    line=dict(
                        color="rgba(255, 0, 0, 0.5)",
                        width=2,
                    ),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                showlegend=True,
                xaxis2_title="Time (frames)",
                yaxis2_title="Pattern Count",
                xaxis1_title="Time (frames)",
                yaxis1_title="Time (frames)"
            )
            
            logger.info("Created pattern map visualization")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern visualization: {str(e)}")
            raise

    def plot_mix_density(self,
                        density: np.ndarray,
                        matches: List[Dict],
                        confidence_scores: np.ndarray,
                        title: str = "Mix Density Analysis") -> go.Figure:
        """
        Create interactive mix density visualization
        
        Args:
            density: Mix density array
            matches: List of detected matches
            confidence_scores: Match confidence scores
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Mix Density Timeline",
                    "Confidence Scores"
                ),
                vertical_spacing=0.1
            )
            
            # Plot mix density
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(density)),
                    y=density,
                    name="Mix Density",
                    mode='lines',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Plot confidence scores
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(confidence_scores)),
                    y=confidence_scores,
                    name="Confidence",
                    mode='lines',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Add match markers
            for match in matches:
                start_time = match['start_time']
                duration = match.get('duration', 1.0)
                confidence = match['confidence']
                
                # Convert time to frame index
                start_frame = int(start_time * self.sr / self.hop_length)
                end_frame = int((start_time + duration) * self.sr / self.hop_length)
                
                # Add markers to both subplots
                for row in [1, 2]:
                    fig.add_shape(
                        type="rect",
                        x0=start_frame,
                        x1=end_frame,
                        y0=0,
                        y1=1,
                        line=dict(
                            color="rgba(255, 0, 0, 0.5)",
                            width=2,
                        ),
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        row=row, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=600,
                showlegend=True,
                xaxis2_title="Time (frames)",
                yaxis2_title="Confidence",
                xaxis1_title="Time (frames)",
                yaxis1_title="Density"
            )
            
            logger.info("Created mix density visualization")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating density visualization: {str(e)}")
            raise
