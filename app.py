import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import torch
import plotly.graph_objects as go

from src.model import TransformerChordRecognizer, CRNNChordBaseline
from src.etl_pipeline import ChordVocabularyMapper

# 1. Page Configuration and Theme
st.set_page_config(
    page_title="AI Chord Progression Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling to enforce dark theme context
st.markdown("""
    <style>
    .main {
        background-color: #111111;
        color: #ffffff;
    }
    .stApp {
        background-color: #111111;
    }
    </style>
    """, unsafe_allow_html=True)

# Compelling Header
st.title("🎵 AI Chord Progression Analyzer")
st.markdown("""
    Upload an audio track (WAV, MP3, FLAC, M4A) to extract its Constant-Q Transform (CQT) features and 
    evaluate its chord progression in real-time.
    
    *Powered by a Transformer model trained on McGill-Billboard annotations.*
""")

# Sidebar settings
st.sidebar.header("🔧 Model Settings")
model_type = st.sidebar.selectbox("Select Model Architecture", ["Transformer", "CRNN"])
confidence_threshold = st.sidebar.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("📁 Audio Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Audio File", 
    type=["wav", "mp3", "m4a", "flac"]
)

# Root colors for visualization (assigned based on chord root notes)
ROOT_COLORS = {
    'C': 'rgba(255, 99, 132, 0.35)',    # Red
    'C#': 'rgba(255, 159, 64, 0.35)',   # Orange
    'D': 'rgba(255, 205, 86, 0.35)',    # Yellow
    'D#': 'rgba(75, 192, 192, 0.35)',   # Teal
    'E': 'rgba(54, 162, 235, 0.35)',    # Blue
    'F': 'rgba(153, 102, 255, 0.35)',   # Purple
    'F#': 'rgba(231, 233, 237, 0.35)',  # Light Grey
    'G': 'rgba(255, 99, 255, 0.35)',    # Magenta
    'G#': 'rgba(100, 255, 100, 0.35)',  # Light Green
    'A': 'rgba(0, 255, 200, 0.35)',     # Cyan
    'A#': 'rgba(139, 69, 19, 0.35)',    # Brown
    'B': 'rgba(218, 112, 214, 0.35)',   # Orchid
    'N': 'rgba(128, 128, 128, 0.15)'    # Neutral Grey
}

# 2. Caching Audio Loading Function
@st.cache_data
def load_audio(file_path):
    """
    Load audio waveform and sample rate (resampled to mono at 22050 Hz).
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        return y, sr
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None

def predict_chords(y, sr, model_type):
    """
    Examines CQT features of the waveform and runs model inference to map chord intervals.
    """
    hop_length = 512
    n_bins = 84
    
    # 1. CQT Extraction
    fmin = librosa.note_to_hz('C1')
    C = librosa.cqt(
        y, 
        sr=sr, 
        hop_length=hop_length, 
        fmin=fmin, 
        n_bins=n_bins, 
        bins_per_octave=12
    )
    cqt_magnitude = np.abs(C) # shape: [84, num_frames]
    
    # 2. Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if model_type == "Transformer":
        model = TransformerChordRecognizer(input_bins=n_bins, num_classes=25)
        checkpoint_path = "models/best_model.pth"
    else:
        model = CRNNChordBaseline(input_bins=n_bins, num_classes=25)
        checkpoint_path = "models/best_model_crnn.pth"
        
    # Load weights if checkpoint exists, otherwise fall back to randomly initialized model
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        st.toast(f"Loaded weights from {checkpoint_path}", icon="✅")
    else:
        st.toast("Model checkpoint not found. Using randomly initialized weights.", icon="⚠️")
        
    model.to(device)
    model.eval()
    
    # 3. Model Inference
    # Convert CQT magnitude to PyTorch float tensor
    cqt_tensor = torch.FloatTensor(cqt_magnitude).unsqueeze(0).to(device) # shape: [1, 84, num_frames]
    
    # --- AUDIO DSP FIX: Log-Compression & Z-Score Normalization ---
    # Convert to float and compress huge linear amplitudes
    cqt_tensor = torch.log1p(cqt_tensor * 10.0)
    
    # Instance-wise Z-Score normalization (Mean=0, Std=1)
    mean = cqt_tensor.mean(dim=(1, 2), keepdim=True)
    std = cqt_tensor.std(dim=(1, 2), keepdim=True) + 1e-8
    cqt_tensor = (cqt_tensor - mean) / std
    # --------------------------------------------------------------
    
    # Model input shape: [batch_size, num_frames, input_bins]
    cqt_tensor = cqt_tensor.permute(0, 2, 1)
    
    with torch.no_grad():
        logits = model(cqt_tensor) # shape: [1, num_frames, 25]
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # shape: [num_frames]
        
    # 4. Group consecutive frames into temporal intervals
    chord_names = [
        'C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
        'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
        'N'
    ]
    
    intervals = []
    current_chord_id = predictions[0]
    start_frame = 0
    
    for i in range(1, len(predictions)):
        if predictions[i] != current_chord_id:
            end_frame = i
            start_time = start_frame * hop_length / sr
            end_time = end_frame * hop_length / sr
            chord_name = chord_names[current_chord_id]
            intervals.append({
                'Start Time (s)': round(start_time, 2),
                'End Time (s)': round(end_time, 2),
                'Chord': chord_name,
                'Root': chord_name.split(':')[0] if ':' in chord_name else chord_name
            })
            start_frame = i
            current_chord_id = predictions[i]
            
    # Add final segment
    end_frame = len(predictions)
    start_time = start_frame * hop_length / sr
    end_time = end_frame * hop_length / sr
    chord_name = chord_names[current_chord_id]
    intervals.append({
        'Start Time (s)': round(start_time, 2),
        'End Time (s)': round(end_time, 2),
        'Chord': chord_name,
        'Root': chord_name.split(':')[0] if ':' in chord_name else chord_name
    })
    
    return pd.DataFrame(intervals)

# 3. App Main Execution
if uploaded_file is not None:
    st.info("Reading uploaded file... Processing may take a few seconds.")
    
    # Save uploaded file to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load audio from cached loader
        y, sr = load_audio(temp_file_path)
        
        if y is not None:
            total_duration = len(y) / sr
            
            # Predict Chords
            df_chords = predict_chords(y, sr, model_type)
            
            # Downsample waveform for Plotly interactive display performance
            downsample = max(1, len(y) // 2500)
            y_downsampled = y[::downsample]
            time_downsampled = np.linspace(0, total_duration, len(y_downsampled))
            
            # 4. Create Plotly Figure (Visual Overlay Layout)
            fig = go.Figure()
            
            # Layer 1: Waveform
            fig.add_trace(go.Scatter(
                x=time_downsampled, 
                y=y_downsampled,
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.45)', width=1.5),
                hoverinfo='skip',
                name='Waveform'
            ))
            
            # Layer 2: Overlay Chord Rectangles (VRECTs)
            for _, row in df_chords.iterrows():
                start = row['Start Time (s)']
                end = row['End Time (s)']
                chord = row['Chord']
                root = row['Root']
                
                # Fetch distinct root color representation
                fill_color = ROOT_COLORS.get(root, 'rgba(120, 120, 120, 0.25)')
                
                # Add shaded overlay boundaries
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=fill_color,
                    opacity=0.6,
                    layer="below",
                    line_width=1,
                    line_color="rgba(255, 255, 255, 0.15)",
                    annotation_text=chord,
                    annotation_position="top left",
                    annotation_font=dict(size=12, color="white", family="Arial Black")
                )
                
            # Chart Aesthetics Configuration
            fig.update_layout(
                title=dict(
                    text="Chord Alignment Timeline overlaying Audio Waveform",
                    x=0.5,
                    font=dict(size=16, color='white', family='Arial Bold')
                ),
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                showlegend=False,
                plot_bgcolor='#111111',
                paper_bgcolor='#111111',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, zeroline=False, range=[0, total_duration]),
                yaxis=dict(showgrid=False, zeroline=False, range=[-1.1, 1.1]),
                margin=dict(l=40, r=40, t=60, b=40),
                height=500
            )
            
            # Render chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Synchronized Audio Playback
            st.subheader("🎵 Audio Playback")
            st.audio(temp_file_path)
            
            # Chord Transitions Data Details
            st.markdown("---")
            with st.expander("📊 Raw Prediction Interval Data"):
                st.dataframe(df_chords, use_container_width=True)
                
    except Exception as e:
        st.error(f"Failed to process file: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
else:
    # Sidebar prompt when no audio file is uploaded
    st.info("👈 Please upload an audio file (.wav, .mp3, etc.) in the sidebar to run the analysis.")
    
    # Showcase layout sample structure
    st.markdown("### 💡 How it works")
    st.markdown("""
        1. **Sidebar Input**: Upload your audio track in the sidebar and choose the model architecture (**Transformer** is recommended).
        2. **Visual Timeline**: The app processes the track and renders a Plotly chart displaying the audio envelope with transparent color blocks overlaid on top representing chord duration.
        3. **Synchronized Playback**: Use the audio player below the plot to listen to the song and follow the chord progression visually.
    """)

# Instructions on running the app
# Run the app locally in the terminal:
# PYTHONPATH=. ./.venv/bin/streamlit run app.py
