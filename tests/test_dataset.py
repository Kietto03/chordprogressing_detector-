import h5py
import numpy as np
import torch
import pytest
from src.dataset import ChordDataset, get_dataloaders


def test_chord_dataset_chunking(temp_h5_long):
    """Test that ChordDataset correctly chunks a long file (e.g. 7031 frames -> floor(/215) chunks)."""
    ds = ChordDataset([temp_h5_long], chunk_length_frames=215)
    # 7031 // 215 = 32 full chunks
    assert len(ds) == 32
    cqt, labels = ds[0]
    assert cqt.shape == (84, 215)
    assert labels.shape == (215,)
    assert isinstance(cqt, torch.FloatTensor)
    assert isinstance(labels, torch.LongTensor)


def test_chord_dataset_short_file(tmp_path):
    """Song shorter than chunk length should yield 0 chunks."""
    path = tmp_path / "short.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("cqt", data=np.random.rand(84, 100).astype(np.float32))
        f.create_dataset("labels", data=np.random.randint(0, 25, 100).astype(np.int32))
    ds = ChordDataset([str(path)], chunk_length_frames=215)
    assert len(ds) == 0


def test_get_dataloaders_song_level_split(temp_h5, temp_h5_long, tmp_path):
    """get_dataloaders must split at song level (not chunk level) and be deterministic with seed."""
    # Put two files in a temp processed dir
    processed = tmp_path / "proc"
    processed.mkdir()
    # copy the fixtures
    import shutil
    shutil.copy(temp_h5, processed / "song1.h5")
    shutil.copy(temp_h5_long, processed / "song2.h5")

    train_loader, val_loader = get_dataloaders(str(processed), batch_size=2, train_split=0.5)
    # With 2 songs, one should go to train, one to val (train_len = 1)
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    # Total chunks should be non-overlapping
    train_files = {f for f, _, _ in train_loader.dataset.index_map}
    val_files = {f for f, _, _ in val_loader.dataset.index_map}
    assert train_files.isdisjoint(val_files)
    assert len(train_files) + len(val_files) == 2
