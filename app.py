import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import torch
import io
import base64
import json
from scipy.io import wavfile
from src.model import TransformerChordRecognizer, CRNNChordBaseline
from src.etl_pipeline import ChordVocabularyMapper
from src.config import (
    CHUNK_LENGTH_FRAMES,
    HOP_SIZE as DEFAULT_HOP_SIZE,
    N_BINS,
    HOP_LENGTH,
    SR as TARGET_SR,
)
try:
    from src.viz import build_chord_timeline_figure
except Exception:
    from viz import build_chord_timeline_figure  # fallback when running as script

# Helper function to load external style sheet cleanly without leakage
def load_css(file_name="src/style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Convert sharp chords to flat chords if flat spelling is preferred
def spell_chord(chord_name, use_flats=True):
    if not chord_name or chord_name == 'N':
        return 'N'
    if not use_flats:
        return chord_name
        
    parts = chord_name.split(':')
    root = parts[0]
    quality = parts[1] if len(parts) > 1 else ''
    
    flat_roots = {
        'C#': 'Db',
        'D#': 'Eb',
        'F#': 'Gb',
        'G#': 'Ab',
        'A#': 'Bb'
    }
    
    new_root = flat_roots.get(root, root)
    if quality:
        return f"{new_root}:{quality}"
    return new_root

# Helper function to clean chord names for human-readable display
def clean_chord_name(chord_name):
    if not chord_name or chord_name == 'N':
        return 'N'
    parts = chord_name.split(':')
    root = parts[0]
    mod = parts[1] if len(parts) > 1 else ''
    
    # Map common modifiers to standard names
    if mod == 'maj':
        suffix = ''
    elif mod == 'min':
        suffix = 'm'
    elif mod == 'maj7':
        suffix = 'maj7'
    elif mod == 'min7':
        suffix = 'm7'
    elif mod == '7':
        suffix = '7'
    elif mod == 'dim':
        suffix = 'dim'
    elif mod == 'aug':
        suffix = 'aug'
    elif mod == 'sus4':
        suffix = 'sus4'
    elif mod == 'sus2':
        suffix = 'sus2'
    else:
        suffix = mod
    return f"{root}{suffix}"

# Estimate Roman Numeral scale degrees based on Key Scale Signature
def get_roman_numeral(chord_name, key_sig):
    if not chord_name or chord_name == 'N' or not key_sig:
        return ""
    
    parts = chord_name.split(':')
    chord_root = parts[0]
    
    # Check if minor quality
    is_major = True
    if len(parts) > 1:
        is_major = 'min' not in parts[1] and 'm' not in parts[1]
    else:
        pitch_len = 2 if (len(chord_root) > 1 and chord_root[1] in ['#', 'b']) else 1
        rest = chord_root[pitch_len:]
        if rest.startswith('m') and not rest.startswith('maj'):
            is_major = False
            
    # Normalize flats to sharps
    flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    chord_root_norm = flats.get(chord_root, chord_root)
    
    key_parts = key_sig.split()
    if len(key_parts) < 2:
        return ""
    key_root = key_parts[0]
    key_mode = key_parts[1].lower() # "major" or "minor"
    key_root_norm = flats.get(key_root, key_root)
    
    chromatic = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if chord_root_norm not in chromatic or key_root_norm not in chromatic:
        return ""
        
    semitones = (chromatic.index(chord_root_norm) - chromatic.index(key_root_norm)) % 12
    
    if key_mode == "minor":
        mapping = {
            0: {True: 'I', False: 'i'},
            1: {True: 'bII', False: 'bii'},
            2: {True: 'II', False: 'ii'},
            3: {True: 'III', False: 'iii'},
            4: {True: 'IV', False: 'iv'},
            5: {True: 'IV', False: 'iv'},
            6: {True: 'bV', False: 'bv'},
            7: {True: 'V', False: 'v'},
            8: {True: 'VI', False: 'vi'},
            9: {True: 'bVII', False: 'bvii'},
            10: {True: 'VII', False: 'vii'},
            11: {True: 'vii°', False: 'vii°'}
        }
    else:
        mapping = {
            0: {True: 'I', False: 'i'},
            1: {True: 'bII', False: 'bii'},
            2: {True: 'II', False: 'ii'},
            3: {True: 'bIII', False: 'biii'},
            4: {True: 'III', False: 'iii'},
            5: {True: 'IV', False: 'iv'},
            6: {True: 'bV', False: 'bv'},
            7: {True: 'V', False: 'v'},
            8: {True: 'bVI', False: 'bvi'},
            9: {True: 'VI', False: 'vi'},
            10: {True: 'bVII', False: 'bvii'},
            11: {True: 'vii°', False: 'vii°'}
        }
        
    return mapping.get(semitones, {}).get(is_major, "")

# Map Chord Suffix to Semitone Intervals relative to root note
def get_chord_intervals(chord_name):
    if not chord_name or chord_name == 'N':
        return [0]
    parts = chord_name.split(':')
    mod = parts[1] if len(parts) > 1 else ''
    
    if 'min7' in mod or 'm7' in mod:
        return [0, 3, 7, 10]
    elif 'maj7' in mod:
        return [0, 4, 7, 11]
    elif '7' in mod:
        return [0, 4, 7, 10]
    elif 'min' in mod or 'm' in mod:
        return [0, 3, 7]
    elif 'dim' in mod:
        return [0, 3, 6]
    elif 'aug' in mod:
        return [0, 4, 8]
    elif 'sus4' in mod:
        return [0, 5, 7]
    elif 'sus2' in mod:
        return [0, 2, 7]
    else:
        return [0, 4, 7] # major triad default

# Translate each chord class into its exact MIDI pitch triad/tetrad matrix
# centered inside the visual range [48, 72] (C3 to C5)
def get_chord_midi_notes(chord_name):
    if not chord_name or chord_name == 'N':
        return []
    parts = chord_name.split(':')
    root = parts[0]
    quality = parts[1] if len(parts) > 1 else 'maj'
    
    chromatic = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    root_norm = flats.get(root, root)
    if root_norm not in chromatic:
        return []
    
    semi = chromatic.index(root_norm)
    # If root is F (5) to B (11), place root in octave 3 (midi 53-59).
    # If root is C (0) to E (4), place root in octave 4 (midi 60-64).
    # This keeps all chord component notes perfectly in [48, 72].
    if semi >= 5:
        root_midi = 48 + semi
    else:
        root_midi = 60 + semi
        
    # Get intervals
    if 'min7' in quality or 'm7' in quality:
        intervals = [0, 3, 7, 10]
    elif 'maj7' in quality:
        intervals = [0, 4, 7, 11]
    elif '7' in quality:
        intervals = [0, 4, 7, 10]
    elif 'min' in quality or 'm' in quality:
        intervals = [0, 3, 7]
    elif 'dim' in quality:
        intervals = [0, 3, 6]
    elif 'aug' in quality:
        intervals = [0, 4, 8]
    elif 'sus4' in quality:
        intervals = [0, 5, 7]
    elif 'sus2' in quality:
        intervals = [0, 2, 7]
    else:
        intervals = [0, 4, 7]
        
    return [root_midi + i for i in intervals]



# Render Interactive SVG Volume Dial
def render_dial(label, value):
    # Angle from -135 to 135 (total 270 degrees)
    angle = -135 + value * 270
    stroke_offset = 138 - (value * 103)
    svg_html = f"""
    <div style="text-align: center; margin: 10px 0;">
        <svg width="60" height="60" viewBox="0 0 60 60">
            <!-- Dial background -->
            <circle cx="30" cy="30" r="22" fill="#1E293B" stroke="#334155" stroke-width="2" />
            <!-- Active Fill ring -->
            <circle cx="30" cy="30" r="22" fill="none" stroke="#2563EB" stroke-width="2" 
                    stroke-dasharray="138" stroke-dashoffset="{stroke_offset}" 
                    transform="rotate(135 30 30)" />
            <!-- Rotating Indicator needle -->
            <line x1="30" y1="30" x2="30" y2="12" stroke="#60A5FA" stroke-width="3" stroke-linecap="round"
                  transform="rotate({angle} 30 30)" />
            <!-- Center hub cap -->
            <circle cx="30" cy="30" r="4" fill="#60A5FA" />
        </svg>
        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; user-select: none;">{label}</div>
    </div>
    """
    return svg_html

# Generate SMF MIDI Type 0 File from predicted chords sequence in pure Python
def create_midi_file(df_chords, tempo_bpm=120):
    ticks_per_sec = (tempo_bpm * 480) / 60.0
    midi_events = []
    
    for _, row in df_chords.iterrows():
        chord = row['Chord']
        if chord == 'N':
            continue
        start_time = row['Start Time (s)']
        end_time = row['End Time (s)']
        
        # Fetch notes from notes mapping
        notes = get_chord_midi_notes(chord)
        
        start_tick = int(start_time * ticks_per_sec)
        end_tick = int(end_time * ticks_per_sec)
        
        for note in notes:
            note = max(0, min(127, note))
            # Note On (Status 0x90, velocity 80)
            midi_events.append((start_tick, 0x90, note, 80))
            # Note Off (Status 0x80, velocity 0)
            midi_events.append((end_tick, 0x80, note, 0))
            
    # Sort events chronologically (Note off events first if identical tick times)
    midi_events.sort(key=lambda x: (x[0], x[1]))
    
    track_data = bytearray()
    
    # Meta Event: Set Tempo (0xFF 0x51 0x03)
    usec_per_qn = int(60000000 / tempo_bpm)
    track_data.extend([0x00, 0xFF, 0x51, 0x03, 
                       (usec_per_qn >> 16) & 0xFF, 
                       (usec_per_qn >> 8) & 0xFF, 
                       usec_per_qn & 0xFF])
                       
    # Variable Length Quantity encoder helper
    def to_vlq(delta):
        if delta == 0:
            return [0]
        bin_str = []
        while delta > 0:
            bin_str.append(delta & 0x7F)
            delta >>= 7
        bin_str.reverse()
        for i in range(len(bin_str) - 1):
            bin_str[i] |= 0x80
        return bin_str
        
    last_tick = 0
    for tick, status, note, vel in midi_events:
        delta = tick - last_tick
        last_tick = tick
        track_data.extend(to_vlq(delta))
        track_data.extend([status, note, vel])
        
    # Meta Event: End of Track (0xFF 0x2F 0x00)
    track_data.extend([0x00, 0xFF, 0x2F, 0x00])
    
    # MIDI Header MThd
    header = bytearray("MThd", "ascii")
    header.extend([0x00, 0x00, 0x00, 0x06])
    header.extend([0x00, 0x00]) # Format 0
    header.extend([0x00, 0x01]) # 1 Track
    header.extend([0x01, 0xE0]) # 480 Ticks per Quarter Note (0x01E0)
    
    # Track Header MTrk
    track = bytearray("MTrk", "ascii")
    track_len = len(track_data)
    track.extend([(track_len >> 24) & 0xFF, 
                  (track_len >> 16) & 0xFF, 
                  (track_len >> 8) & 0xFF, 
                  track_len & 0xFF])
    track.extend(track_data)
    
    return bytes(header + track)



def y_to_wav_bytes(y, sr=22050):
    y_scaled = np.clip(y, -1.0, 1.0)
    y_int16 = (y_scaled * 32767).astype(np.int16)
    wav_bytes_io = io.BytesIO()
    wavfile.write(wav_bytes_io, sr, y_int16)
    return wav_bytes_io.getvalue()

def to_base64_str(audio_bytes):
    return base64.b64encode(audio_bytes).decode('utf-8')


# 1. Page Configuration (Wide Layout Enforced)
st.set_page_config(
    page_title="AI Chord Progression Analyzer & Transcriber",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Outfit and Inter Google Fonts
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Inject styling rules from dedicated file
load_css("src/style.css")

# Initialize session state variables
if 'audio_vol' not in st.session_state:
    st.session_state.audio_vol = 0.8
if 'playing' not in st.session_state:
    st.session_state.playing = False

# 2. Hero Banner Header Injection
st.markdown(
    '<div class="hero-container">'
    '<h1 class="hero-title">🎵 AI Chord Progression Analyzer & Transcriber</h1>'
    '<p class="hero-subtitle">Academic Defense Demonstration: Real-Time Audio Chord Transcription & Spectro-Temporal Alignment using Transformer-Attention Architecture</p>'
    '</div>',
    unsafe_allow_html=True
)

# 3. Sidebar Control Card Layout
st.sidebar.markdown(
    '<div style="margin-bottom:1.5rem;">'
    '<h3 style="font-family:Outfit,sans-serif;font-weight:700;margin:0;color:#0F172A;">🔧 Control panel</h3>'
    '<p style="font-size:0.85rem;color:#64748B;margin:0;">Configure pipeline execution settings</p>'
    '</div>',
    unsafe_allow_html=True
)

model_type = st.sidebar.selectbox(
    "Select Model Architecture", 
    ["Transformer", "CRNN"], 
    help="Transformer leverages global attention and positional encoding, while the CRNN baseline uses convolutional layers and RNN recurrence."
)

# Early check for model checkpoint
chk_path = "models/best_model.pth" if model_type == "Transformer" else "models/best_model_crnn.pth"
if not os.path.exists(chk_path):
    st.sidebar.error(f"⚠️ Checkpoint `{chk_path}` not found! Please run the training script `train.py` for this variant first.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='font-family:Outfit,sans-serif;font-weight:700;color:#0F172A;'>📁 File Upload Zone</h4>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Choose a track...", 
    type=["wav", "mp3", "m4a", "flac"]
)

# Key & Tempo Estimation algorithms
def estimate_tempo_and_key(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    tempo = int(round(float(tempo)))
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    major_profile = (major_profile - major_profile.mean()) / major_profile.std()
    minor_profile = (minor_profile - minor_profile.mean()) / minor_profile.std()
    chroma_norm = (chroma_mean - chroma_mean.mean()) / (chroma_mean.std() + 1e-8)
    
    major_corrs = [np.corrcoef(chroma_norm, np.roll(major_profile, i))[0, 1] for i in range(12)]
    minor_corrs = [np.corrcoef(chroma_norm, np.roll(minor_profile, i))[0, 1] for i in range(12)]
    
    best_maj_idx = np.argmax(major_corrs)
    best_min_idx = np.argmax(minor_corrs)
    
    if major_corrs[best_maj_idx] > minor_corrs[best_min_idx]:
        key = f"{pitches[best_maj_idx]} Major"
    else:
        key = f"{pitches[best_min_idx]} Minor"
        
    # Normalize sharp key signatures to standard flat spellings where appropriate
    flat_key_map = {
        'A# Major': 'Bb Major',
        'A# Minor': 'Bb Minor',
        'D# Major': 'Eb Major',
        'D# Minor': 'Eb Minor',
        'G# Major': 'Ab Major',
        'G# Minor': 'Ab Minor',
        'C# Major': 'Db Major',
        'F# Major': 'Gb Major',
    }
    key = flat_key_map.get(key, key)
    return tempo, key


@st.cache_data
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file: {e}")


def predict_chords(y, sr, model_type, status_drawer):
    hop_length = HOP_LENGTH
    n_bins = N_BINS
    chunk_length = CHUNK_LENGTH_FRAMES  # Align sequence length with training!
    hop_size = DEFAULT_HOP_SIZE         # 50% overlap for stable prediction
    
    # 1. CQT Extraction
    status_drawer.update(label="🧬 Step 2: Extracting Log-CQT Features...", state="running")
    fmin = librosa.note_to_hz('C1')
    C = librosa.cqt(
        y, 
        sr=sr, 
        hop_length=hop_length, 
        fmin=fmin, 
        n_bins=n_bins, 
        bins_per_octave=12
    )
    cqt_magnitude = np.abs(C)
    
    # 2. Model Setup
    status_drawer.update(label=f"🧠 Step 3: Loading {model_type} Neural Model Weights...", state="running")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if model_type == "Transformer":
        model = TransformerChordRecognizer(input_bins=n_bins, num_classes=25)
        checkpoint_path = "models/best_model.pth"
    else:
        model = CRNNChordBaseline(input_bins=n_bins, num_classes=25)
        checkpoint_path = "models/best_model_crnn.pth"
        
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        st.error(f"❌ Critical Error: Checkpoint not found at {checkpoint_path}. Execution halted.")
        st.stop()
        
    model.to(device)
    model.eval()
    
    # Convert CQT matrix to PyTorch Tensor and apply log compression
    cqt_tensor = torch.FloatTensor(cqt_magnitude).unsqueeze(0).to(device) # shape: [1, 84, total_frames]
    cqt_tensor = torch.log1p(cqt_tensor * 10.0)
    total_frames = cqt_tensor.size(2)
    
    status_drawer.update(label="🧠 Step 4: Computing Model Predictions...", state="running")
    
    logits_sum = torch.zeros(1, total_frames, 25, device=device)
    logits_count = torch.zeros(1, total_frames, 25, device=device)
    
    # Sliding window inference using exact sequence lengths and chunk-wise normalization
    start_idx = 0
    while start_idx < total_frames:
        end_idx = min(start_idx + chunk_length, total_frames)
        actual_chunk_len = end_idx - start_idx
        
        if actual_chunk_len < chunk_length:
            if total_frames >= chunk_length:
                start_idx = total_frames - chunk_length
                end_idx = total_frames
                actual_chunk_len = chunk_length
            
        chunk = cqt_tensor[:, :, start_idx:end_idx]
        
        if actual_chunk_len < chunk_length:
            # Pad if the entire song is shorter than 215 frames
            pad_len = chunk_length - actual_chunk_len
            padding = torch.zeros(1, n_bins, pad_len, device=device)
            chunk = torch.cat([chunk, padding], dim=2)
            
        # Instance-wise Z-score normalization matching training exactly
        mean = chunk.mean(dim=(1, 2), keepdim=True)
        std = chunk.std(dim=(1, 2), keepdim=True) + 1e-8
        chunk_norm = (chunk - mean) / std
        
        # Permute to shape [1, chunk_length, n_bins]
        chunk_input = chunk_norm.permute(0, 2, 1)
        
        with torch.no_grad():
            chunk_logits = model(chunk_input) # shape: [1, chunk_length, 25]
            
        if actual_chunk_len < chunk_length and total_frames < chunk_length:
            logits_sum[:, :total_frames, :] += chunk_logits[:, :total_frames, :]
            logits_count[:, :total_frames, :] += 1.0
            break
        elif start_idx == total_frames - chunk_length:
            logits_sum[:, start_idx:end_idx, :] += chunk_logits[:, :, :]
            logits_count[:, start_idx:end_idx, :] += 1.0
            break
        else:
            logits_sum[:, start_idx:end_idx, :] += chunk_logits[:, :actual_chunk_len, :]
            logits_count[:, start_idx:end_idx, :] += 1.0
            start_idx += hop_size
            
    logits = logits_sum / (logits_count + 1e-8)
    predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
    
    chord_names = [
        'C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
        'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
        'N'
    ]
    
    # 5. Beat Tracking & Quantization
    status_drawer.update(label="🥁 Step 5: Detecting Musical Beats & Tempo...", state="running")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    
    # Ensure full coverage of the timeline
    total_duration = len(y) / sr
    beat_times = np.concatenate(([0.0], beat_times, [total_duration]))
    if len(beat_times) <= 2:
        # Fallback to a 0.5-second grid if no beats detected
        beat_times = np.arange(0.0, total_duration, 0.5)
        if beat_times[-1] < total_duration:
            beat_times = np.append(beat_times, total_duration)
            
    # Quantize predictions to beat intervals using majority vote
    beat_predictions = []
    for i in range(len(beat_times) - 1):
        t_start = beat_times[i]
        t_end = beat_times[i+1]
        
        # Map time boundary to frame index boundary
        frame_start = int(np.round(t_start * sr / hop_length))
        frame_end = int(np.round(t_end * sr / hop_length))
        
        frame_start = max(0, min(frame_start, len(predictions) - 1))
        frame_end = max(frame_start + 1, min(frame_end, len(predictions)))
        
        interval_preds = predictions[frame_start:frame_end]
        if len(interval_preds) > 0:
            vals, counts = np.unique(interval_preds, return_counts=True)
            dominant_chord_id = vals[np.argmax(counts)]
        else:
            dominant_chord_id = predictions[frame_start]
            
        beat_predictions.append(dominant_chord_id)
        
    chord_names = [
        'C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
        'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
        'N'
    ]
    
    # 6. Event-Driven Compression
    compressed_segments = []
    if len(beat_predictions) > 0:
        current_chord_id = beat_predictions[0]
        start_idx = 0
        
        for i in range(1, len(beat_predictions)):
            if beat_predictions[i] != current_chord_id:
                end_idx = i
                start_time = beat_times[start_idx]
                end_time = beat_times[end_idx]
                chord_name = chord_names[current_chord_id]
                compressed_segments.append({
                    'Start Time (s)': round(start_time, 2),
                    'End Time (s)': round(end_time, 2),
                    'Start Beat': start_idx,
                    'End Beat': end_idx,
                    'Chord': chord_name,
                    'Root': chord_name.split(':')[0] if ':' in chord_name else chord_name,
                    'Duration (Beats)': end_idx - start_idx
                })
                current_chord_id = beat_predictions[i]
                start_idx = i
                
        # Append the final segment
        end_idx = len(beat_predictions)
        start_time = beat_times[start_idx]
        end_time = beat_times[end_idx]
        chord_name = chord_names[current_chord_id]
        compressed_segments.append({
            'Start Time (s)': round(start_time, 2),
            'End Time (s)': round(end_time, 2),
            'Start Beat': start_idx,
            'End Beat': end_idx,
            'Chord': chord_name,
            'Root': chord_name.split(':')[0] if ':' in chord_name else chord_name,
            'Duration (Beats)': end_idx - start_idx
        })
        
    return pd.DataFrame(compressed_segments), beat_times


# Main App Process Trigger
if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        file_key = f"{uploaded_file.name}_{model_type}"
        if ('processed_file_key' not in st.session_state or 
            st.session_state.processed_file_key != file_key):
            
            # Unified tracking Status Element Drawer
            status_element = st.status("🔄 Step 1: Loading Audio Track...", expanded=True)
            
            try:
                y, sr = load_audio(temp_file_path)
                
                if y is not None:
                    total_duration = len(y) / sr
                    
                    # Format total duration to MM:SS
                    duration_minutes = int(total_duration // 60)
                    duration_seconds = int(total_duration % 60)
                    duration_formatted = f"{duration_minutes:02d}:{duration_seconds:02d}"
                    
                    # Run Key & Tempo Estimation
                    status_element.update(label="🧬 Estimating Musical Key & Tempo...", state="running")
                    tempo_bpm, key_signature = estimate_tempo_and_key(y, sr)
                    
                    # Execute CQT + Sliding Window Inference
                    df_chords, beat_times = predict_chords(y, sr, model_type, status_element)
                    
                    # Pre-process Chord names & Roman numerals
                    flat_keys = {
                        'F Major', 'Bb Major', 'Eb Major', 'Ab Major', 'Db Major', 'Gb Major',
                        'D Minor', 'G Minor', 'C Minor', 'F Minor', 'Bb Minor', 'Eb Minor'
                    }
                    use_flats = key_signature in flat_keys
                    if use_flats:
                        df_chords['Chord'] = df_chords['Chord'].apply(lambda c: spell_chord(c, True))
                        
                    df_chords['Chord_Clean'] = df_chords['Chord'].apply(clean_chord_name)
                    df_chords['Roman'] = df_chords.apply(lambda r: get_roman_numeral(r['Chord'], key_signature), axis=1)
                    
                    # Store variables in session state to prevent reprocessing
                    st.session_state.y = y
                    st.session_state.sr = sr
                    st.session_state.total_duration = total_duration
                    st.session_state.duration_formatted = duration_formatted
                    st.session_state.tempo_bpm = tempo_bpm
                    st.session_state.key_signature = key_signature
                    st.session_state.df_chords = df_chords
                    st.session_state.beat_times = beat_times
                    st.session_state.processed_file_key = file_key
                    
                    # Complete the status drawer
                    status_element.update(label="✅ Success: Chord Transcribing & Synthesis Complete!", state="complete", expanded=False)
            except Exception as analysis_err:
                status_element.update(label="❌ Analysis Failed", state="error")
                st.error(f"Error analyzing audio: {str(analysis_err)}")
                st.stop()
        
        # Load values from session state
        y = st.session_state.y
        sr = st.session_state.sr
        total_duration = st.session_state.total_duration
        duration_formatted = st.session_state.duration_formatted
        tempo_bpm = st.session_state.tempo_bpm
        key_signature = st.session_state.key_signature
        df_chords = st.session_state.df_chords

        beat_times = st.session_state.beat_times
        
        # Compute dominant chord
        most_frequent = df_chords['Chord'].mode()[0] if not df_chords.empty else "N"
        most_frequent_clean = clean_chord_name(most_frequent)
        
        # Consolidate metadata badges inside sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h4 style='font-family:Outfit,sans-serif;font-weight:700;color:#0F172A;'>📊 Track Profile & Key</h4>", unsafe_allow_html=True)
        badges_html = (
            f'<div class="sidebar-badge-container">'
            f'<div class="sidebar-badge">'
            f'<span class="badge-label">⏱️ Duration</span>'
            f'<span class="badge-value">{duration_formatted}</span>'
            f'</div>'
            f'<div class="sidebar-badge">'
            f'<span class="badge-label">🥁 Tempo</span>'
            f'<span class="badge-value">{tempo_bpm} BPM</span>'
            f'</div>'
            f'<div class="sidebar-badge">'
            f'<span class="badge-label">🔑 Key Signature</span>'
            f'<span class="badge-value">{key_signature}</span>'
            f'</div>'
            f'<div class="sidebar-badge">'
            f'<span class="badge-label">🎯 Dominant Chord</span>'
            f'<span class="badge-value">{most_frequent_clean}</span>'
            f'</div>'
            f'</div>'
        )
        st.sidebar.markdown(badges_html, unsafe_allow_html=True)
        
        # PR4 SAFE UI: native audio only (no custom iframe controls, no parent DOM access)
        with st.sidebar:
            st.audio(y, sample_rate=sr)

        # Safe Plotly timeline + manual scrub (replaces all old beat-grid + JS sync)
        st.markdown("<h3 class='section-title'>🎵 Chord Timeline (scrub to preview)</h3>", unsafe_allow_html=True)

        if "scrub_time" not in st.session_state:
            st.session_state.scrub_time = 0.0

        scrub_time = st.slider("Scrub time (s)", 0.0, float(total_duration), float(st.session_state.scrub_time), 0.05, key="scrub_slider")
        st.session_state.scrub_time = scrub_time

        fig = build_chord_timeline_figure(y, sr, df_chords, scrub_time=scrub_time)
        st.plotly_chart(fig, use_container_width=True, key="chord_timeline")

        # Live badge
        cur = df_chords[(df_chords["Start Time (s)"] <= scrub_time) & (df_chords["End Time (s)"] > scrub_time)]
        if not cur.empty:
            r = cur.iloc[0]
            st.markdown(f"<div style='font-size:1.25rem;font-weight:800;color:#7928CA;padding:3px 8px;background:#F1E7FF;border-radius:6px;display:inline-block;'>Now: <b>{r.get('Chord_Clean', r.get('Chord','N'))}</b> {r.get('Roman','')}</div>", unsafe_allow_html=True)
        else:
            st.caption("N / silence")

        # Safe click-to-scrub (limited buttons, pure Streamlit)
        st.markdown("**Beat nav (click to jump)**")
        mb = min(24, len(beat_times)-1)
        if mb > 0:
            bcols = st.columns(min(8, mb))
            for b in range(mb):
                with bcols[b % 8]:
                    if st.button(str(b), key=f"bnav{b}"):
                        st.session_state.scrub_time = float(beat_times[b])
                        st.rerun()

        # Bottom controls and utilities
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Convert Another Song", key="convert_again_btn"):
                st.session_state.clear()
                st.rerun()
                
        with col_btn2:
            midi_data = create_midi_file(df_chords, tempo_bpm=tempo_bpm)
            st.download_button(
                label="📥 Download Transcribed MIDI File",
                data=midi_data,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.mid",
                mime="audio/midi",
                key="download_midi_btn"
            )
            
        st.markdown("---")
        
        # Hide raw frame registry inside expander
        with st.expander("📋 View Analytical Chord Registry Data"):
            st.dataframe(
                df_chords[['Start Time (s)', 'End Time (s)', 'Start Beat', 'End Beat', 'Chord_Clean', 'Roman']], 
                use_container_width=True
            )
            
    except Exception as general_err:
        st.error(f"Failed to process file: {str(general_err)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
else:
    # Sidebar prompt layout card
    st.info("👈 Please upload an audio file (.wav, .mp3, etc.) in the sidebar to run the analysis.")
    
    instructions_html = (
        '<div style="background: white; border: 1px solid #E2E8F0; padding: 2rem; border-radius: 20px; box-shadow: 0 8px 24px rgba(148,163,184,0.02); margin-top: 1rem;">'
        '<h3 style="font-family: Outfit, sans-serif; font-weight: 700; margin-top: 0; color: #0F172A;">💡 System Workflow Instructions</h3>'
        '<ol style="margin-bottom: 0; padding-left: 1.25rem; color: #334155;">'
        '<li style="margin-bottom: 0.75rem;"><strong>Parameters</strong>: Adjust model architectures using the panel in the sidebar.</li>'
        '<li style="margin-bottom: 0.75rem;"><strong>Upload</strong>: Drag & drop your target audio file into the zone.</li>'
        '<li style="margin-bottom: 0.75rem;"><strong>Transcription</strong>: The system executes feature extraction, key & tempo analysis, and sequence inference inside a sliding-window frame to ensure compatibility.</li>'
        '<li style="margin-bottom: 0.75rem;"><strong>Analysis</strong>: Observe chord overlays aligned directly onto the audio wave visualization.</li>'
        '</ol>'
        '</div>'
    )
    st.markdown(instructions_html, unsafe_allow_html=True)

