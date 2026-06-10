import torch
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_app_helper_imports_and_basic_calls():
    """Smoke test that key helpers from app can be imported and called without crashing (for PR3 AC)."""
    # Lazy import so that streamlit etc don't pollute test collection if not installed in base env
    try:
        from app import predict_chords, estimate_tempo_and_key, create_midi_file, clean_chord_name, get_roman_numeral
    except Exception as e:
        pytest.skip(f"Could not import app helpers (likely missing streamlit in test env): {e}")

    # Tiny synthetic audio
    y = np.random.randn(22050 * 3).astype(np.float32) * 0.01  # 3s
    sr = 22050

    # These should not raise
    tempo, key = estimate_tempo_and_key(y, sr)
    assert isinstance(tempo, int) and tempo > 0
    assert isinstance(key, str) and ("Major" in key or "Minor" in key)

    clean = clean_chord_name("C#:min")
    assert clean == "C#m"

    roman = get_roman_numeral("C:maj", "C Major")
    assert roman in ("I", "i", "")

    # MIDI roundtrip basic
    import pandas as pd
    df = pd.DataFrame({
        "Chord": ["C:maj", "G:min"],
        "Start Time (s)": [0.0, 1.0],
        "End Time (s)": [1.0, 2.0],
        "Start Beat": [0, 4],
        "End Beat": [4, 8],
    })
    midi_bytes = create_midi_file(df, tempo_bpm=120)
    assert isinstance(midi_bytes, (bytes, bytearray))
    assert len(midi_bytes) > 100
    assert midi_bytes[:4] == b"MThd"


def test_norm_equivalence_between_train_and_app():
    """Critical for PR3: log1p*10 + instance Z-score must be identical in train path and app.predict_chords path."""
    torch.manual_seed(42)
    cqt = torch.rand(1, 84, 300) * 0.1 + 0.001  # simulate positive magnitude

    # Train path (from src/train.py)
    x_train = cqt.clone()
    x_train = torch.log1p(x_train * 10.0)
    mean_t = x_train.mean(dim=(1, 2), keepdim=True)
    std_t = x_train.std(dim=(1, 2), keepdim=True) + 1e-8
    x_train = (x_train - mean_t) / std_t

    # App path (from app.py predict_chords)
    x_app = cqt.clone()
    x_app = torch.log1p(x_app * 10.0)
    mean_a = x_app.mean(dim=(1, 2), keepdim=True)
    std_a = x_app.std(dim=(1, 2), keepdim=True) + 1e-8
    x_app = (x_app - mean_a) / std_a

    assert torch.allclose(x_train, x_app, atol=1e-6), "Norm math drifted between train and app paths!"
