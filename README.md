# Sample Detection Demo

This project implements a music sample detection system that can find occurrences of a short audio sample within longer songs. It uses state-of-the-art audio processing techniques including:
- Demucs for audio source separation
- Basic Pitch for MIDI conversion
- Feature extraction and pattern matching

## Installation

1. Clone the repository with submodules:
```bash
git clone https://github.com/yourusername/musicdna.git
cd musicdna
git submodule update --init --recursive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Demo

1. Start the Gradio interface:
```bash
python app.py
```

2. Open the web interface (default: http://localhost:7860)

## Usage

1. Upload a short audio sample (5-15 seconds)
2. Upload 3-5 songs to search through
3. View results showing:
   - Match locations with confidence scores
   - Matched audio segments
   - Similarity visualizations

## Project Structure

- `sample_detector.py`: Core implementation of sample detection algorithm
- `app.py`: Gradio web interface
- `requirements.txt`: Project dependencies

## Notes

- The sample should be 5-15 seconds long for optimal results
- Supported audio formats: WAV, MP3, FLAC
- GPU acceleration is used if available
