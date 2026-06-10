import pandas as pd
import numpy as np
import pytest
from src.etl_pipeline import AudioETLPipeline


def test_align_labels_to_frames_basic():
    pipeline = AudioETLPipeline(target_sr=22050, hop_length=512, n_bins=84)
    # Simple 3-chord example
    df = pd.DataFrame({
        "start_time": [0.0, 1.0, 3.0],
        "end_time": [1.0, 3.0, 5.0],
        "chord": ["C:maj", "G:min", "N"]
    })
    # ~ 5s at 22050 / 512 ≈ 215 frames per second? Wait, frames = time * sr / hop
    num_frames = int(5 * 22050 / 512) + 5  # a bit over
    labels = pipeline.align_labels_to_frames(df, num_frames)
    assert labels[0] == 0          # C:maj
    assert labels[50] == 19        # G:min (roughly at 1s)
    assert labels[-5] == 24        # N
    assert 0 <= labels.min() <= labels.max() <= 24


def test_parse_labels_and_align_edge_clipping():
    pipeline = AudioETLPipeline()
    df = pd.DataFrame({
        "start_time": [-1.0, 10.0],
        "end_time": [0.5, 99.0],
        "chord": ["C:maj", "D:min"]
    })
    labels = pipeline.align_labels_to_frames(df, num_frames=100)
    assert labels[0] == 0
    assert labels[99] == 14  # D:min (clipped)
    assert labels.dtype == np.int32
