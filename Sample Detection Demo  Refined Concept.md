## Sample Detection Demo

**Core Implementation**
```python
def find_sample_in_songs(sample, songs):
    # Stem Separation
    sample_stems = demucs_separate(sample)
    songs_stems = [demucs_separate(song) for song in songs]
    
    # MIDI Conversion
    sample_midi = basic_pitch_convert(sample_stems)
    songs_midi = [basic_pitch_convert(song_stems) for song_stems in songs_stems]
    
    # Feature Extraction
    sample_features = extract_features(sample_stems, sample_midi)
    
    # Pattern Matching
    matches = []
    for song_idx, (song_stems, song_midi) in enumerate(zip(songs_stems, songs_midi)):
        song_features = extract_features(song_stems, song_midi)
        similarity = compute_similarity(sample_features, song_features)
        
        if similarity > threshold:
            matches.append({
                "song_index": song_idx,
                "timestamp": found_at_timestamp,
                "confidence": similarity,
                "matched_features": matched_feature_details
            })
    
    return sorted(matches, key=lambda x: x["confidence"], reverse=True)
```

**Gradio Interface**
```python
import gradio as gr

def demo_interface(sample_audio, comparison_audio_list):
    results = find_sample_in_songs(sample_audio, comparison_audio_list)
    return format_results(results)

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
    ]
)
```

**Key Features**
- Upload and process short samples (5-15 seconds)
- Compare against 3-5 full songs
- Utilize existing demucs and basic pitch pipeline
- Display match locations with confidence scores
- Visualize matching segments
- Play matched segments side by side
- Show similarity metrics