"""
Visualization module for MusicDNA.
Shows stem decomposition and dual fingerprinting (spectrogram + MIDI).
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import pretty_midi
from basic_pitch.inference import predict

class MusicDNAVisualizer:
    def __init__(self):
        self.colors = {
            'drums': '#FF9900',  # Orange
            'bass': '#3366CC',   # Blue
            'vocals': '#DC3912', # Red
            'other': '#109618'   # Green
        }
        
    def plot_stem_decomposition(self, stems: Dict[str, np.ndarray], 
                              sample_rate: int = 44100,
                              highlight_region: Optional[tuple] = None):
        """
        Visualize stem decomposition with optional sample highlight.
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(4, 1, figure=fig)
        
        for idx, (stem_name, audio) in enumerate(stems.items()):
            ax = fig.add_subplot(gs[idx])
            
            # Plot waveform
            times = np.arange(len(audio)) / sample_rate
            ax.plot(times, audio, color=self.colors[stem_name], linewidth=1)
            
            # Highlight sample region if specified
            if highlight_region:
                start, end = highlight_region
                ax.axvspan(start, end, color='yellow', alpha=0.3)
            
            # Customize appearance
            ax.set_ylabel(stem_name.capitalize())
            ax.grid(True, alpha=0.3)
            
            # Only show x-axis for bottom plot
            if idx < 3:
                ax.set_xticks([])
        
        ax.set_xlabel('Time (s)')
        plt.tight_layout()
        return fig
        
    def plot_dual_fingerprint(self, audio: np.ndarray, sample_rate: int = 44100):
        """
        Show both spectrogram and MIDI representation of audio.
        """
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 1, figure=fig)
        
        # Spectrogram (Visual DNA)
        ax1 = fig.add_subplot(gs[0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time',
                                     sr=sample_rate, ax=ax1)
        ax1.set_title('Visual DNA (Spectrogram)')
        fig.colorbar(img, ax=ax1, format='%+2.0f dB')
        
        # MIDI representation (MIDI DNA)
        ax2 = fig.add_subplot(gs[1])
        midi_data = predict(audio, sample_rate)
        times = np.arange(len(audio)) / sample_rate
        
        # Plot MIDI notes
        for note in midi_data['pitch_list']:
            time, pitch, conf = note
            ax2.scatter(time, pitch, c='blue', alpha=conf, s=50)
            
        ax2.set_title('MIDI DNA (Note Pattern)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('MIDI Note')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_match_analysis(self, sample: np.ndarray, 
                          match_location: Dict,
                          stems: Dict[str, np.ndarray],
                          sample_rate: int = 44100):
        """
        Visualize a sample match in context.
        """
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(5, 1, figure=fig)
        
        # Plot sample
        ax_sample = fig.add_subplot(gs[0])
        times_sample = np.arange(len(sample)) / sample_rate
        ax_sample.plot(times_sample, sample, color='black')
        ax_sample.set_title('Sample')
        ax_sample.set_xticks([])
        
        # Plot stems with match highlight
        start_time = match_location['start_time'] / sample_rate
        end_time = start_time + len(sample) / sample_rate
        
        for idx, (stem_name, audio) in enumerate(stems.items()):
            ax = fig.add_subplot(gs[idx + 1])
            times = np.arange(len(audio)) / sample_rate
            ax.plot(times, audio, color=self.colors[stem_name])
            
            # Highlight match if in this stem
            if stem_name == match_location['stem']:
                ax.axvspan(start_time, end_time, 
                          color='yellow', alpha=0.3,
                          label=f'Match (conf: {match_location["confidence"]:.2f})')
                ax.legend()
            
            ax.set_ylabel(stem_name.capitalize())
            if idx < 3:
                ax.set_xticks([])
        
        ax.set_xlabel('Time (s)')
        plt.tight_layout()
        return fig
        
    def save_visualization(self, fig, filename: str):
        """Save visualization to file."""
        fig.savefig(filename, dpi=300, bbox_inches='tight')
