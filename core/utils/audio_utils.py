"""Audio processing utilities for MusicDNA."""
import numpy as np
import librosa
import torch
from typing import Tuple, Optional, Dict, Any

def load_audio(file_path: str, sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Load audio file with error handling."""
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {str(e)}")

def extract_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Extract comprehensive audio features."""
    features = {}
    
    # Temporal features
    features['tempo'], _ = librosa.beat.beat_track(y=audio, sr=sr)
    features['onset_strength'] = librosa.onset.onset_strength(y=audio, sr=sr)
    features['rms'] = librosa.feature.rms(y=audio)[0]
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Spectral features
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # Mel features
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    features['mel_spectrogram'] = librosa.power_to_db(mel_spec)
    
    return features

def compute_spectrograms(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Compute multiple spectrogram representations."""
    specs = {}
    
    # STFT
    specs['stft'] = np.abs(librosa.stft(audio))
    
    # Mel spectrogram
    specs['mel'] = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    # Chroma
    specs['chroma'] = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    return specs

def apply_gpu_acceleration(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to GPU if available."""
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    return audio / np.max(np.abs(audio))

def segment_audio(audio: np.ndarray, segment_length: int, 
                 hop_length: Optional[int] = None) -> np.ndarray:
    """Segment audio into overlapping windows."""
    if hop_length is None:
        hop_length = segment_length // 2
        
    segments = librosa.util.frame(audio, frame_length=segment_length, 
                                hop_length=hop_length)
    return segments

def cache_features(func):
    """Decorator for caching computed features."""
    cache = {}
    
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper
