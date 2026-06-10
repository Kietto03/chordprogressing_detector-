import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.config import CHUNK_LENGTH_FRAMES as DEFAULT_CHUNK_LENGTH, DEFAULT_SEED


class ChordDataset(Dataset):
    """
    PyTorch Dataset that loads CQT features and aligned chord labels from HDF5 files.
    Chops long audio signals into fixed-size chunks of length `chunk_length_frames`.
    """
    def __init__(self, processed_dir_or_files, chunk_length_frames=DEFAULT_CHUNK_LENGTH):
        self.chunk_length_frames = chunk_length_frames
        
        # Locate all .h5 files
        if isinstance(processed_dir_or_files, list):
            self.file_paths = processed_dir_or_files
        else:
            self.processed_dir = processed_dir_or_files
            self.file_paths = []
            if os.path.exists(self.processed_dir):
                self.file_paths = [
                    os.path.join(self.processed_dir, f) 
                    for f in os.listdir(self.processed_dir) 
                    if f.lower().endswith('.h5')
                ]
        self.file_paths.sort()
        
        # Build the index map: idx -> (file_path, start_frame, end_frame)
        self.index_map = []
        for file_path in self.file_paths:
            try:
                # Open HDF5 file only to read the shape of the datasets (very efficient!)
                with h5py.File(file_path, 'r') as f:
                    # cqt has shape (n_bins, n_frames)
                    num_frames = f['cqt'].shape[1]
                    
                # Calculate how many full chunks of chunk_length_frames fit into this song
                num_chunks = num_frames // self.chunk_length_frames
                for chunk_idx in range(num_chunks):
                    start_frame = chunk_idx * self.chunk_length_frames
                    end_frame = start_frame + self.chunk_length_frames
                    self.index_map.append((file_path, start_frame, end_frame))
            except Exception as e:
                print(f"Warning: Failed to read metadata from {file_path}. Skipping. Error: {e}")

    def __len__(self):
        """
        Return the total number of valid chunks across all songs.
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Fetch a single chunk. Open file, extract the frame slice, and convert to tensors.
        
        Note: Opening and caching the HDF5 file handles inside __getitem__ per process worker 
        is safe for PyTorch multi-process DataLoader workers, and avoids excessive I/O overhead.
        """
        file_path, start_frame, end_frame = self.index_map[idx]
        
        try:
            if not hasattr(self, 'files'):
                self.files = {}
            if file_path not in self.files:
                self.files[file_path] = h5py.File(file_path, 'r')
            f = self.files[file_path]
            
            # Slice the datasets directly from disk
            cqt_slice = f['cqt'][:, start_frame:end_frame]
            label_slice = f['labels'][start_frame:end_frame]
                
            # Convert to PyTorch Tensors
            cqt_tensor = torch.FloatTensor(cqt_slice)
            label_tensor = torch.LongTensor(label_slice)
            
            return cqt_tensor, label_tensor
        except Exception as e:
            raise RuntimeError(f"Error loading index {idx} (file: {file_path}, frames: {start_frame}-{end_frame}): {e}")


def get_dataloaders(processed_dir, batch_size=32, train_split=0.8, seed: int = DEFAULT_SEED):
    """
    Split the HDF5 song files into train and validation sets at the song level,
    returning standard PyTorch DataLoaders.

    seed: controls the song-level shuffle for reproducibility (PR5).
    """
    if not os.path.exists(processed_dir):
        raise ValueError(f"Processed directory {processed_dir} does not exist.")
        
    # 1. Locate all HDF5 files
    file_paths = [
        os.path.join(processed_dir, f) 
        for f in os.listdir(processed_dir) 
        if f.lower().endswith('.h5')
    ]
    file_paths.sort()
    
    total_songs = len(file_paths)
    if total_songs == 0:
        raise ValueError(f"No valid .h5 files found in {processed_dir}")
        
    # 2. Split at the song level using the provided seed for reproducibility (PR5)
    rng = np.random.default_rng(seed)
    shuffled_paths = list(file_paths)
    rng.shuffle(shuffled_paths)
    
    train_len = int(train_split * total_songs)
    # Ensure at least 1 validation song if there are multiple songs
    if train_len == total_songs and total_songs > 1:
        train_len = total_songs - 1
        
    train_files = shuffled_paths[:train_len]
    val_files = shuffled_paths[train_len:]
    
    # 3. Instantiate the datasets using file splits
    train_dataset = ChordDataset(train_files)
    val_dataset = ChordDataset(val_files)
    
    print(f"Song-level dataset split: {len(train_files)} training songs, {len(val_files)} validation songs.")
    print(f"Total training chunks: {len(train_dataset)}, validation chunks: {len(val_dataset)}")
    
    # 4. Create DataLoader instances
    # Using num_workers=0 to ensure maximum compatibility across platforms
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    processed_dir = 'data/processed'
    print(f"Testing dataloader module with processed_dir: {processed_dir}")
    
    try:
        train_loader, val_loader = get_dataloaders(processed_dir, batch_size=4, train_split=0.8)
        print(f"Total training batches: {len(train_loader)}")
        print(f"Total validation batches: {len(val_loader)}")
        
        # Fetch a single batch
        X_batch, y_batch = next(iter(train_loader))
        
        print("\nVerification successful!")
        print(f"X_batch (CQT) shape: {X_batch.shape}")
        print(f"y_batch (Labels) shape: {y_batch.shape}")
        
    except Exception as e:
        print(f"Error occurred during verification: {e}", file=sys.stderr)
        sys.exit(1)
