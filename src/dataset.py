import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ChordDataset(Dataset):
    """
    PyTorch Dataset that loads CQT features and aligned chord labels from HDF5 files.
    Chops long audio signals into fixed-size chunks of length `chunk_length_frames`.
    """
    def __init__(self, processed_dir, chunk_length_frames=215):
        self.processed_dir = processed_dir
        self.chunk_length_frames = chunk_length_frames
        
        # Locate all .h5 files
        self.file_paths = []
        if os.path.exists(processed_dir):
            self.file_paths = [
                os.path.join(processed_dir, f) 
                for f in os.listdir(processed_dir) 
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
        
        Note: Opening and closing the HDF5 file handle inside __getitem__ is fork-safe
        and standard practice for PyTorch multi-process DataLoader workers.
        """
        file_path, start_frame, end_frame = self.index_map[idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Slice the datasets directly from disk
                cqt_slice = f['cqt'][:, start_frame:end_frame]
                label_slice = f['labels'][start_frame:end_frame]
                
            # Convert to PyTorch Tensors
            cqt_tensor = torch.FloatTensor(cqt_slice)
            label_tensor = torch.LongTensor(label_slice)
            
            return cqt_tensor, label_tensor
        except Exception as e:
            raise RuntimeError(f"Error loading index {idx} (file: {file_path}, frames: {start_frame}-{end_frame}): {e}")


def get_dataloaders(processed_dir, batch_size=32, train_split=0.8):
    """
    Split the ChordDataset into train and validation sets, returning standard PyTorch DataLoaders.
    """
    # 1. Instantiate the dataset
    dataset = ChordDataset(processed_dir)
    total_len = len(dataset)
    
    if total_len == 0:
        raise ValueError(f"No valid chunks of size {dataset.chunk_length_frames} found in {processed_dir}")
        
    # 2. Divide into train and validation sets
    train_len = int(train_split * total_len)
    val_len = total_len - train_len
    
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    
    # 3. Create DataLoader instances
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
