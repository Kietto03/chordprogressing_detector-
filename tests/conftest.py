import pytest
import h5py
import numpy as np
import torch
import tempfile
import os
from src.etl_pipeline import ChordVocabularyMapper
from src.model import TransformerChordRecognizer, CRNNChordBaseline


@pytest.fixture
def mapper():
    """Fixture for the chord vocabulary mapper."""
    return ChordVocabularyMapper()


@pytest.fixture
def temp_h5(tmp_path):
    """Create a temporary small HDF5 file mimicking processed data (full-song style)."""
    path = tmp_path / "test_song.h5"
    n_bins = 84
    n_frames = 500  # small for tests
    with h5py.File(path, "w") as f:
        cqt = np.random.rand(n_bins, n_frames).astype(np.float32) * 0.1
        labels = np.random.randint(0, 25, n_frames).astype(np.int32)
        # Make a few segments more realistic
        labels[100:200] = 0  # C:maj
        labels[200:250] = 12  # C:min
        f.create_dataset("cqt", data=cqt, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    return str(path)


@pytest.fixture
def temp_h5_long(tmp_path):
    """Larger temp h5 for chunking tests (e.g. 7031 frames like real data)."""
    path = tmp_path / "test_long.h5"
    n_bins = 84
    n_frames = 7031
    with h5py.File(path, "w") as f:
        cqt = np.random.rand(n_bins, n_frames).astype(np.float32) * 0.05
        labels = np.random.randint(0, 25, n_frames).astype(np.int32)
        f.create_dataset("cqt", data=cqt, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
    return str(path)


@pytest.fixture
def tiny_cqt():
    """Small synthetic CQT tensor for norm / inference smoke tests."""
    return torch.rand(1, 84, 300) * 0.1  # [batch, bins, frames] style before permute in places


@pytest.fixture
def mapper_instance():
    return ChordVocabularyMapper()


@pytest.fixture
def small_synthetic_batch():
    """Small batch for train smoke."""
    batch_size = 2
    seq_len = 215
    n_bins = 84
    X = torch.rand(batch_size, n_bins, seq_len)
    y = torch.randint(0, 25, (batch_size, seq_len))
    return X, y
