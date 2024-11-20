import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class DemucsProcessor:
    def __init__(self):
        logger.info("Initializing Demucs processor")
        self.model = get_model('htdemucs')
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        logger.info("Demucs model loaded successfully")

    def separate_stems(self, audio_path):
        """Separate audio into stems using Demucs"""
        logger.info(f"Processing audio file: {audio_path}")
        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            wav = wav.mean(0, keepdim=True)  # Convert to mono
            
            # Move to GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            wav = wav.to(device)
            
            # Apply model
            with torch.no_grad():
                stems = apply_model(self.model, wav, shifts=1, split=True, overlap=0.25)
                stems = stems.cpu().numpy()
            
            # Create dictionary with stems
            stem_dict = {
                name: stems[i] for i, name in enumerate(['drums', 'bass', 'other', 'vocals'])
            }
            
            logger.info("Stem separation completed successfully")
            return stem_dict, sr
            
        except Exception as e:
            logger.error(f"Error in stem separation: {str(e)}")
            raise
