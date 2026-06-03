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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.model import TransformerChordRecognizer, CRNNChordBaseline
from src.etl_pipeline import ChordVocabularyMapper

# Helper function to load external style sheet cleanly without leakage
def load_css(file_name="src/style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# Assign Chord octave center pitches dynamically using Voice Leading Heuristic
def assign_chord_pitches(df_chords):
    chromatic = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    
    pitches = []
    prev_pitch = 24 # Start at C2 (pitch index 24)
    
    for _, row in df_chords.iterrows():
        root = row['Root']
        root_norm = flats.get(root, root)
        if root_norm not in chromatic:
            pitches.append(prev_pitch)
            continue
            
        semi = chromatic.index(root_norm)
        
        # Select octave from [1, 2, 3] to minimize vertical pitch jump
        best_pitch = prev_pitch
        min_dist = 999
        for octave in [1, 2, 3]:
            pitch = octave * 12 + semi
            dist = abs(pitch - prev_pitch)
            if dist < min_dist:
                min_dist = dist
                best_pitch = pitch
                
        pitches.append(best_pitch)
        prev_pitch = best_pitch
        
    df_chords['Pitch'] = pitches
    return df_chords

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

# Task 3: Synthesize guide track containing pure sine waves + harmonics matching the note-level map
def synthesize_chords_guide(df_chords, duration, sr=22050):
    num_samples = int(duration * sr)
    audio_data = np.zeros(num_samples, dtype=np.float32)
    
    # MIDI note to frequency helper
    def m2f(m):
        return 440.0 * (2.0 ** ((m - 69.0) / 12.0))
        
    for _, row in df_chords.iterrows():
        start_time = row['Start Time (s)']
        end_time = row['End Time (s)']
        chord = row['Chord']
        
        if chord == 'N':
            continue
            
        notes = get_chord_midi_notes(chord)
        
        start_sample = int(start_time * sr)
        end_sample = min(num_samples, int(end_time * sr))
        t = np.arange(end_sample - start_sample) / sr
        
        chord_wave = np.zeros(end_sample - start_sample, dtype=np.float32)
        for note in notes:
            freq = m2f(note)
            # Create a simple rich organ sound: fundamental + 1st harmonic
            wave = 0.6 * np.sin(2.0 * np.pi * freq * t) + 0.3 * np.sin(4.0 * np.pi * freq * t)
            chord_wave += wave
            
        if len(notes) > 0:
            chord_wave = chord_wave / len(notes)
            
        # Apply fade-in and fade-out to prevent pops/clicks
        fade_len = min(int(0.02 * sr), len(chord_wave) // 2)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len)
            fade_out = np.linspace(1.0, 0.0, fade_len)
            chord_wave[:fade_len] *= fade_in
            chord_wave[-fade_len:] *= fade_out
            
        audio_data[start_sample:end_sample] = chord_wave
        
    # Scale to 16-bit integer PCM WAV format bytes
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    wav_bytes_io = io.BytesIO()
    wavfile.write(wav_bytes_io, sr, audio_int16)
    return wav_bytes_io.getvalue()

# Mix backing audio and chord guide audio in Python on the backend
def mix_audio_tracks(backing_y, guide_wav_bytes, backing_vol, guide_vol, sr=22050):
    try:
        if guide_wav_bytes is None:
            return y_to_wav_bytes(backing_y, sr)
            
        guide_sr, guide_data = wavfile.read(io.BytesIO(guide_wav_bytes))
        guide_y = guide_data.astype(np.float32) / 32767.0
        
        length = min(len(backing_y), len(guide_y))
        mixed = (backing_y[:length] * backing_vol) + (guide_y[:length] * guide_vol)
        
        # Normalize to prevent digital clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
            
        mixed_int16 = (mixed * 32767).astype(np.int16)
        wav_bytes_io = io.BytesIO()
        wavfile.write(wav_bytes_io, sr, mixed_int16)
        return wav_bytes_io.getvalue()
    except Exception as e:
        # Safe fallback: return raw backing audio WAV bytes
        try:
            return y_to_wav_bytes(backing_y, sr)
        except Exception:
            return b""

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
if 'chord_vol' not in st.session_state:
    st.session_state.chord_vol = 0.5
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'guide_wav' not in st.session_state:
    st.session_state.guide_wav = None

# 2. Hero Banner Header Injection
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🎵 AI Chord Progression Analyzer & Transcriber</h1>
        <p class="hero-subtitle">Academic Defense Demonstration: Real-Time Audio Chord Transcription & Spectro-Temporal Alignment using Transformer-Attention Architecture</p>
    </div>
    """, unsafe_allow_html=True)

# 3. Sidebar Control Card Layout
st.sidebar.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <h3 style='font-family:Outfit,sans-serif;font-weight:700;margin:0;color:#0F172A;'>🔧 Control panel</h3>
        <p style='font-size:0.85rem;color:#64748B;margin:0;'>Configure pipeline execution settings</p>
    </div>
    """, unsafe_allow_html=True)

model_type = st.sidebar.selectbox(
    "Select Model Architecture", 
    ["Transformer", "CRNN"], 
    help="Transformer leverages global attention and positional encoding, while the CRNN baseline uses convolutional layers and RNN recurrence."
)

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
        
    return tempo, key


@st.cache_data
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file: {e}")


def predict_chords(y, sr, model_type, status_drawer):
    hop_length = 512
    n_bins = 84
    chunk_length = 5000
    hop_size = 2500 # 50% overlap
    
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
        st.toast(f"No checkpoint found at {checkpoint_path}. Running default fallback parameters.", icon="⚠️")
        
    model.to(device)
    model.eval()
    
    # 3. Normalization
    cqt_tensor = torch.FloatTensor(cqt_magnitude).unsqueeze(0).to(device)
    cqt_tensor = torch.log1p(cqt_tensor * 10.0)
    mean = cqt_tensor.mean(dim=(1, 2), keepdim=True)
    std = cqt_tensor.std(dim=(1, 2), keepdim=True) + 1e-8
    cqt_tensor = (cqt_tensor - mean) / std
    
    cqt_tensor = cqt_tensor.permute(0, 2, 1)
    total_frames = cqt_tensor.size(1)
    
    status_drawer.update(label="🧠 Step 4: Computing Transformer Attention Matrix...", state="running")
    
    logits_sum = torch.zeros(1, total_frames, 25, device=device)
    logits_count = torch.zeros(1, total_frames, 25, device=device)
    
    if total_frames <= chunk_length:
        pad_len = chunk_length - total_frames
        if pad_len > 0:
            padding_tensor = torch.zeros(1, pad_len, n_bins, device=device)
            cqt_input = torch.cat([cqt_tensor, padding_tensor], dim=1)
        else:
            cqt_input = cqt_tensor
            
        with torch.no_grad():
            chunk_logits = model(cqt_input)
        logits = chunk_logits[:, :total_frames, :]
    else:
        start_idx = 0
        while start_idx < total_frames:
            if start_idx + chunk_length <= total_frames:
                chunk = cqt_tensor[:, start_idx : start_idx + chunk_length, :]
                with torch.no_grad():
                    chunk_logits = model(chunk)
                logits_sum[:, start_idx : start_idx + chunk_length, :] += chunk_logits
                logits_count[:, start_idx : start_idx + chunk_length, :] += 1.0
                start_idx += hop_size
            else:
                start_idx_last = total_frames - chunk_length
                chunk = cqt_tensor[:, start_idx_last : total_frames, :]
                with torch.no_grad():
                    chunk_logits = model(chunk)
                logits_sum[:, start_idx_last : total_frames, :] += chunk_logits
                logits_count[:, start_idx_last : total_frames, :] += 1.0
                break
                
        logits = logits_sum / (logits_count + 1e-8)
        
    predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
    
    # Apply Majority Vote Smoothing with a smaller window size to preserve short passing chords
    window_size = 5
    half_w = window_size // 2
    n_frames = len(predictions)
    smoothed_predictions = np.copy(predictions)
    
    for idx in range(n_frames):
        start_w = max(0, idx - half_w)
        end_w = min(n_frames, idx + half_w + 1)
        vals, counts = np.unique(predictions[start_w:end_w], return_counts=True)
        smoothed_predictions[idx] = vals[np.argmax(counts)]
        
    predictions = smoothed_predictions
    
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
    
    # Strict Segment Compactor: combine slots and merge segments under 0.1 seconds
    compacted = []
    for seg in intervals:
        duration = seg['End Time (s)'] - seg['Start Time (s)']
        if duration >= 0.1:
            compacted.append(seg)
        else:
            if compacted:
                compacted[-1]['End Time (s)'] = seg['End Time (s)']
            else:
                compacted.append(seg)
                
    # Final contiguous merge sweep
    final_segments = []
    for seg in compacted:
        if final_segments and final_segments[-1]['Chord'] == seg['Chord']:
            final_segments[-1]['End Time (s)'] = seg['End Time (s)']
        else:
            final_segments.append(seg)
            
    return pd.DataFrame(final_segments)


# Widescreen Plotly Timeline Figure (Waveform + Segment blocks)
def create_timeline_plot(y, sr, total_duration, df_chords):
    # 1. Downsample waveform
    downsample_factor = max(1, len(y) // 2000)
    y_down = y[::downsample_factor]
    time_down = np.linspace(0, total_duration, len(y_down))
    
    # Create subplots: Row 1 (Waveform), Row 2 (Chords Timeline)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.3, 0.7]
    )
    
    # Add Waveform envelope (Row 1)
    fig.add_trace(
        go.Scatter(
            x=time_down, 
            y=y_down, 
            line=dict(color='#64748B', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(148, 163, 184, 0.15)',
            hoverinfo='skip',
            name="Waveform"
        ),
        row=1, col=1
    )
    
    # Set Axes parameters
    fig.update_yaxes(range=[0, 1], showgrid=False, showticklabels=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='#E2E8F0', row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='#E2E8F0', row=1, col=1)
    
    # Beautiful color palette for chord roots
    root_colors = {
        'C': '#EF4444', 'C#': '#F97316', 'D': '#F59E0B', 'D#': '#EAB308',
        'E': '#10B981', 'F': '#14B8A6', 'F#': '#06B6D4', 'G': '#3B82F6',
        'G#': '#6366F1', 'A': '#8B5CF6', 'A#': '#A855F7', 'B': '#EC4899',
        'N': '#94A3B8'
    }
    
    # Add rectangles for chord segments in Row 2
    for _, row in df_chords.iterrows():
        start = row['Start Time (s)']
        end = row['End Time (s)']
        chord = row['Chord']
        
        # Parse root chord character for color mapping
        root = chord.split(':')[0] if ':' in chord else chord
        # Clean root name
        root = root.replace('min', '').replace('maj', '').replace('dim', '').replace('aug', '').replace('7', '').strip()
        color = root_colors.get(root, '#94A3B8')
        
        # Add background shape for the chord block
        fig.add_shape(
            type="rect",
            xref="x2", yref="y2",
            x0=start, y0=0.1,
            x1=end, y1=0.9,
            fillcolor=color,
            line=dict(color="#FFFFFF", width=1.5),
            layer="below"
        )
        
        # Add single centered text annotation
        duration = end - start
        if duration > 0.3:
            fig.add_annotation(
                xref="x2", yref="y2",
                x=(start + end) / 2,
                y=0.5,
                text=f"<b>{clean_chord_name(chord)}</b>",
                showarrow=False,
                font=dict(size=12, color="white", family="Outfit, Inter"),
                align="center"
            )
            
    # Set modern light styling for Plotly layout
    fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=20, r=20, t=10, b=20),
        showlegend=False,
        height=350,
        dragmode=False
    )
    
    return fig


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
                    df_chords = predict_chords(y, sr, model_type, status_element)
                    
                    # Synthesize guide track once
                    status_element.update(label="🎹 Synthesizing Harmonic Guide Track via NumPy Oscillators...", state="running")
                    df_chords['Chord_Clean'] = df_chords['Chord'].apply(clean_chord_name)
                    df_chords['Roman'] = df_chords.apply(lambda r: get_roman_numeral(r['Chord'], key_signature), axis=1)
                    df_chords = assign_chord_pitches(df_chords)
                    
                    # Cache synthesized chords guide track in session state
                    guide_wav = synthesize_chords_guide(df_chords, total_duration, sr)
                    
                    # Store variables in session state to prevent reprocessing
                    st.session_state.y = y
                    st.session_state.sr = sr
                    st.session_state.total_duration = total_duration
                    st.session_state.duration_formatted = duration_formatted
                    st.session_state.tempo_bpm = tempo_bpm
                    st.session_state.key_signature = key_signature
                    st.session_state.df_chords = df_chords
                    st.session_state.guide_wav = guide_wav
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
        guide_wav = st.session_state.guide_wav
        
        # Compute dominant chord
        most_frequent = df_chords['Chord'].mode()[0] if not df_chords.empty else "N"
        most_frequent_clean = clean_chord_name(most_frequent)
        
        # Consolidate metadata badges inside sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h4 style='font-family:Outfit,sans-serif;font-weight:700;color:#0F172A;'>📊 Track Profile & Key</h4>", unsafe_allow_html=True)
        st.sidebar.markdown(f"""
            <div class="sidebar-badge-container">
                <div class="sidebar-badge">
                    <span class="badge-label">⏱️ Duration</span>
                    <span class="badge-value">{duration_formatted}</span>
                </div>
                <div class="sidebar-badge">
                    <span class="badge-label">🥁 Tempo</span>
                    <span class="badge-value">{tempo_bpm} BPM</span>
                </div>
                <div class="sidebar-badge">
                    <span class="badge-label">🔑 Key Signature</span>
                    <span class="badge-value">{key_signature}</span>
                </div>
                <div class="sidebar-badge">
                    <span class="badge-label">🎯 Dominant Chord</span>
                    <span class="badge-value">{most_frequent_clean}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Render Sidebar Control Pod HTML5 Iframe
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h4 style='font-family:Outfit,sans-serif;font-weight:700;color:#0F172A;'>🎛️ Volume & Playback Controls</h4>", unsafe_allow_html=True)
        
        backing_wav = y_to_wav_bytes(y, sr)
        backing_b64 = to_base64_str(backing_wav)
        guide_b64 = to_base64_str(guide_wav)
        
        # Format chords list to JSON with MIDI notes
        chords_list = []
        for _, row in df_chords.iterrows():
            start = row['Start Time (s)']
            end = row['End Time (s)']
            chord = row['Chord_Clean']
            roman = row['Roman']
            chords_list.append({
                "chord": chord,
                "roman": roman,
                "start": start,
                "end": end
            })
        chords_json = json.dumps(chords_list)
        
        html_controls = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background-color: transparent;
                    font-family: 'Inter', sans-serif;
                    color: #1E293B;
                    overflow: hidden;
                }
                .control-card {
                    background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-radius: 12px;
                    padding: 14px;
                    box-shadow: 0 4px 12px rgba(148, 163, 184, 0.05);
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                .btn-play {
                    background: linear-gradient(135deg, #7928CA 0%, #00DFD8 100%);
                    color: #FFFFFF;
                    border: none;
                    border-radius: 50%;
                    width: 52px;
                    height: 52px;
                    font-size: 18px;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                    box-shadow: 0 4px 12px rgba(121, 40, 202, 0.2);
                    outline: none;
                }
                .btn-play:hover {
                    transform: scale(1.05);
                    box-shadow: 0 6px 16px rgba(121, 40, 202, 0.35);
                }
                .btn-play:active {
                    transform: scale(0.98);
                }
                .slider-group {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                .slider-label {
                    font-size: 0.72rem;
                    font-weight: 700;
                    color: #64748B;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    display: flex;
                    justify-content: space-between;
                }
                input[type=range] {
                    -webkit-appearance: none;
                    width: 100%;
                    background: transparent;
                    margin: 0;
                }
                input[type=range]:focus {
                    outline: none;
                }
                input[type=range]::-webkit-slider-runnable-track {
                    width: 100%;
                    height: 6px;
                    cursor: pointer;
                    background: #E2E8F0;
                    border-radius: 3px;
                }
                input[type=range]::-webkit-slider-thumb {
                    height: 14px;
                    width: 14px;
                    border-radius: 50%;
                    background: #7928CA;
                    cursor: pointer;
                    -webkit-appearance: none;
                    margin-top: -4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: background 0.15s;
                }
                input[type=range]::-webkit-slider-thumb:hover {
                    background: #00DFD8;
                }
                .active-chord-display {
                    font-size: 1.4rem;
                    font-weight: 800;
                    color: #7928CA;
                    text-align: center;
                    padding: 8px;
                    background: #F8FAFC;
                    border-radius: 8px;
                    border: 1px solid #E2E8F0;
                    font-family: 'Outfit', sans-serif;
                    letter-spacing: 0.05em;
                    min-height: 32px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-shadow: 0 1px 2px rgba(121, 40, 202, 0.05);
                    user-select: none;
                }
            </style>
        </head>
        <body>
            <div class="control-card">
                <button class="btn-play" id="playBtn">▶</button>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Audio Vol</span>
                        <span id="audioVolVal">80%</span>
                    </div>
                    <input type="range" id="audioVol" min="0" max="1" step="0.01" value="0.8">
                </div>
                
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Chord Vol</span>
                        <span id="chordVolVal">50%</span>
                    </div>
                    <input type="range" id="chordVol" min="0" max="1" step="0.01" value="0.5">
                </div>
                
                <div class="active-chord-display" id="activeChord">NO CHORD</div>
                
                <audio id="backingAudio" src="data:audio/wav;base64,__BACKING_B64__"></audio>
                <audio id="guideAudio" src="data:audio/wav;base64,__GUIDE_B64__"></audio>
            </div>

            <script>
                const backingAudio = document.getElementById('backingAudio');
                const guideAudio = document.getElementById('guideAudio');
                const playBtn = document.getElementById('playBtn');
                const audioVol = document.getElementById('audioVol');
                const chordVol = document.getElementById('chordVol');
                const audioVolVal = document.getElementById('audioVolVal');
                const chordVolVal = document.getElementById('chordVolVal');
                const activeChordEl = document.getElementById('activeChord');

                // Parse serialized chords data from python
                const chords = __CHORDS_JSON__;
                const duration = __DURATION__;

                // Sync volumes initially
                backingAudio.volume = parseFloat(audioVol.value);
                guideAudio.volume = parseFloat(chordVol.value);

                // Listeners for volume change
                audioVol.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    backingAudio.volume = val;
                    audioVolVal.textContent = Math.round(val * 100) + '%';
                });

                chordVol.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    guideAudio.volume = val;
                    chordVolVal.textContent = Math.round(val * 100) + '%';
                });

                // Play / Pause toggle function
                function togglePlay() {
                    if (backingAudio.paused) {
                        backingAudio.play().catch(err => console.log("Backing play error", err));
                        guideAudio.currentTime = backingAudio.currentTime;
                        guideAudio.play().catch(err => console.log("Guide play error", err));
                        playBtn.textContent = '⏸';
                    } else {
                        backingAudio.pause();
                        guideAudio.pause();
                        playBtn.textContent = '▶';
                    }
                }

                playBtn.addEventListener('click', togglePlay);

                // Synchronize times on play and seeking
                backingAudio.addEventListener('play', () => {
                    guideAudio.currentTime = backingAudio.currentTime;
                    guideAudio.play().catch(err => console.log("Guide audio play deferred", err));
                    playBtn.textContent = '⏸';
                });

                backingAudio.addEventListener('pause', () => {
                    guideAudio.pause();
                    playBtn.textContent = '▶';
                });

                backingAudio.addEventListener('seeking', () => {
                    guideAudio.currentTime = backingAudio.currentTime;
                });
                backingAudio.addEventListener('seeked', () => {
                    guideAudio.currentTime = backingAudio.currentTime;
                });
                backingAudio.addEventListener('ratechange', () => {
                    guideAudio.playbackRate = backingAudio.playbackRate;
                });

                // Global Spacebar shortcut inside this iframe and parent window
                function setupSpacebar() {
                    const handleSpacebar = (e) => {
                        if (e.code === 'Space') {
                            const activeEl = document.activeElement;
                            if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.isContentEditable)) {
                                return;
                            }
                            e.preventDefault();
                            togglePlay();
                        }
                    };
                    
                    window.addEventListener('keydown', handleSpacebar);
                    
                    try {
                        const parentWindow = window.parent;
                        if (parentWindow) {
                            parentWindow.addEventListener('keydown', handleSpacebar);
                        }
                    } catch (err) {
                        console.log("Could not bind spacebar to parent window:", err);
                    }
                }
                setupSpacebar();

                // Playhead & Active Chord animation loop targeting parent window Plotly chart
                function drawPlayhead() {
                    try {
                        const parentDoc = window.parent.document;
                        const wrapper = parentDoc.getElementById('timeline-wrapper');
                        if (wrapper) {
                            let playhead = parentDoc.getElementById('plotly-playhead');
                            if (!playhead) {
                                playhead = parentDoc.createElement('div');
                                playhead.id = 'plotly-playhead';
                                playhead.style.position = 'absolute';
                                playhead.style.width = '2px';
                                playhead.style.backgroundColor = '#EF4444';
                                playhead.style.pointerEvents = 'none';
                                playhead.style.zIndex = '100';
                                
                                const dot = parentDoc.createElement('div');
                                dot.style.position = 'absolute';
                                dot.style.top = '-4px';
                                dot.style.left = '-3px';
                                dot.style.width = '8px';
                                dot.style.height = '8px';
                                dot.style.borderRadius = '50%';
                                dot.style.backgroundColor = '#EF4444';
                                playhead.appendChild(dot);
                                
                                wrapper.appendChild(playhead);
                            }
                            
                            const gd = wrapper.querySelector('.js-plotly-plot');
                            if (gd && gd._fullLayout && gd._fullLayout.xaxis) {
                                const curTime = backingAudio.currentTime;
                                const xaxis = gd._fullLayout.xaxis;
                                const margin = gd._fullLayout.margin;
                                const xPixel = xaxis.l2p(curTime);
                                
                                if (xPixel !== undefined && !isNaN(xPixel)) {
                                    const left = xPixel + margin.l;
                                    playhead.style.left = left + 'px';
                                    playhead.style.top = margin.t + 'px';
                                    playhead.style.height = (gd._fullLayout.height - margin.t - margin.b) + 'px';
                                    playhead.style.display = 'block';
                                } else {
                                    playhead.style.display = 'none';
                                }
                            } else {
                                playhead.style.display = 'none';
                            }
                        }
                    } catch (err) {
                        // Suppress cross-origin errors
                    }

                    // Update Active Chord Name
                    try {
                        const curTime = backingAudio.currentTime;
                        let active = null;
                        for (let i = 0; i < chords.length; i++) {
                            if (curTime >= chords[i].start && curTime <= chords[i].end) {
                                active = chords[i];
                                break;
                            }
                        }
                        if (activeChordEl) {
                            if (active && active.chord !== 'N') {
                                const romanPart = active.roman ? ` (${active.roman})` : '';
                                activeChordEl.textContent = active.chord + romanPart;
                                activeChordEl.style.color = '#7928CA';
                            } else {
                                activeChordEl.textContent = 'NO CHORD';
                                activeChordEl.style.color = '#64748B';
                            }
                        }
                    } catch (err) {
                        console.log(err);
                    }

                    requestAnimationFrame(drawPlayhead);
                }
                requestAnimationFrame(drawPlayhead);

                // Parent click listener to seek playhead
                function setupParentClickListener() {
                    try {
                        const parentDoc = window.parent.document;
                        const wrapper = parentDoc.getElementById('timeline-wrapper');
                        if (wrapper) {
                            if (wrapper.dataset.clickBound === 'true') return;
                            
                            const handlePlotClick = (e) => {
                                const gd = wrapper.querySelector('.js-plotly-plot');
                                if (gd && gd._fullLayout && gd._fullLayout.xaxis) {
                                    const rect = gd.getBoundingClientRect();
                                    const clickX = e.clientX - rect.left;
                                    const margin = gd._fullLayout.margin;
                                    const plotWidth = gd._fullLayout.width - margin.l - margin.r;
                                    const relativeX = clickX - margin.l;
                                    
                                    if (relativeX >= 0 && relativeX <= plotWidth) {
                                        const pct = relativeX / plotWidth;
                                        const seekTime = pct * duration;
                                        backingAudio.currentTime = seekTime;
                                        guideAudio.currentTime = seekTime;
                                    }
                                }
                            };
                            
                            wrapper.addEventListener('click', handlePlotClick);
                            wrapper.dataset.clickBound = 'true';
                        }
                    } catch (err) {
                        console.log("Could not bind click listener to parent plot:", err);
                    }
                }

                // Retry binding click listener until Plotly container is ready
                let retries = 0;
                const intervalId = setInterval(() => {
                    setupParentClickListener();
                    retries++;
                    if (retries > 30) clearInterval(intervalId);
                }, 100);
            </script>
        </body>
        </html>
        """
        
        html_controls = html_controls.replace("__BACKING_B64__", backing_b64)
        html_controls = html_controls.replace("__GUIDE_B64__", guide_b64)
        html_controls = html_controls.replace("__CHORDS_JSON__", chords_json)
        html_controls = html_controls.replace("__DURATION__", str(total_duration))
        
        # Render the custom controls inside st.sidebar
        with st.sidebar:
            st.components.v1.html(html_controls, height=270)
        
        # Main area timeline visualization
        st.markdown("<h3 class='section-title'>📈 Spectro-Temporal Alignment Timeline</h3>", unsafe_allow_html=True)
        
        # Wrap Plotly figure inside canvas-card with timeline-wrapper id
        st.markdown('<div id="timeline-wrapper" class="canvas-card" style="position: relative;">', unsafe_allow_html=True)
        fig = create_timeline_plot(y, sr, total_duration, df_chords)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        with st.expander("📋 View Frame-by-Frame Chord Index Registry"):
            st.dataframe(
                df_chords[['Start Time (s)', 'End Time (s)', 'Chord_Clean', 'Roman']], 
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
    
    st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; padding: 2rem; border-radius: 20px; box-shadow: 0 8px 24px rgba(148,163,184,0.02); margin-top: 1rem;">
            <h3 style="font-family: Outfit, sans-serif; font-weight: 700; margin-top: 0; color: #0F172A;">💡 System Workflow Instructions</h3>
            <ol style="margin-bottom: 0; padding-left: 1.25rem; color: #334155;">
                <li style="margin-bottom: 0.75rem;"><strong>Parameters</strong>: Adjust model architectures using the panel in the sidebar.</li>
                <li style="margin-bottom: 0.75rem;"><strong>Upload</strong>: Drag & drop your target audio file into the zone.</li>
                <li style="margin-bottom: 0.75rem;"><strong>Transcription</strong>: The system executes feature extraction, key & tempo analysis, and sequence inference inside a sliding-window frame to ensure compatibility.</li>
                <li style="margin-bottom: 0.75rem;"><strong>Analysis</strong>: Observe chord overlays aligned directly onto the audio wave visualization.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
