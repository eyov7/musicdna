import gradio as gr
from sample_detector import SampleDetector
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def create_visualization(sample_features, match_features):
    """Create visualization comparing sample and matched segment"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot sample features
    librosa.display.specshow(sample_features['stems']['drums']['chroma'], 
                           y_axis='chroma', x_axis='time', ax=ax1)
    ax1.set_title('Sample Chromagram')
    
    # Plot matched features
    librosa.display.specshow(match_features['stems']['drums']['chroma'], 
                           y_axis='chroma', x_axis='time', ax=ax2)
    ax2.set_title('Matched Segment Chromagram')
    
    plt.tight_layout()
    return fig

def demo_interface(sample_audio, comparison_audio_list):
    detector = SampleDetector()
    
    # Process inputs
    results = detector.find_sample_in_songs(sample_audio, comparison_audio_list)
    
    if not results:
        return {"message": "No matches found"}, None, None
    
    # Get best match
    best_match = results[0]
    
    # Create visualization
    match_audio = comparison_audio_list[best_match["song_index"]]
    # TODO: Extract matched segment audio
    
    # Format results for display
    formatted_results = {
        "matches": [{
            "song": f"Song {match['song_index'] + 1}",
            "timestamp": f"{match['timestamp']:.2f}s",
            "confidence": f"{match['confidence']:.2%}"
        } for match in results]
    }
    
    # Create visualization
    # TODO: Get actual features for visualization
    sample_features = {}
    match_features = {}
    #viz = create_visualization(sample_features, match_features)
    
    return formatted_results, None, None  # Replace None with actual audio and visualization

# Create Gradio interface
demo = gr.Interface(
    fn=demo_interface,
    inputs=[
        gr.Audio(label="Upload Sample (5-15 seconds)"),
        gr.File(label="Upload Songs to Search (3-5 songs)", file_count="multiple")
    ],
    outputs=[
        gr.JSON(label="Match Results"),
        gr.Audio(label="Matched Segment"),
        gr.Plot(label="Similarity Visualization")
    ],
    title="Sample Detection Demo",
    description="Upload a short sample and find where it appears in other songs."
)

if __name__ == "__main__":
    demo.launch()
