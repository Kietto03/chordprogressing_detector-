import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to sys.path to allow direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import TransformerChordRecognizer, CRNNChordBaseline

class FocalLoss(nn.Module):
    """
    Focal Loss helps combat extreme class imbalance by down-weighting the loss 
    assigned to well-classified examples (like C:maj or N) and focusing on hard examples.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs shape: [batch_size, num_classes, seq_len]
        # targets shape: [batch_size, seq_len]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_model(processed_dir='data/processed', epochs=50, batch_size=32, lr=1e-4, model_type='transformer'):
    """
    Train either CRNN or Transformer model on CQT frame features.
    """
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # 2. Get DataLoaders
    try:
        train_loader, val_loader = get_dataloaders(processed_dir, batch_size=batch_size, train_split=0.8)
        print(f"Loaded datasets. Training chunks: {len(train_loader.dataset)}, Validation chunks: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    # 3. Data Sanity Check
    print("\n=== DATA SANITY CHECK ===")
    sanity_batch = next(iter(train_loader))
    X_sanity, y_sanity = sanity_batch
    print(f"Sanity Batch - X_batch (CQT) shape: {X_sanity.shape}")
    print(f"Sanity Batch - y_batch (Labels) shape: {y_sanity.shape}")
    
    unique_vals, count_vals = np.unique(y_sanity.numpy(), return_counts=True)
    print("Sanity Batch - Label Histogram (Unique label IDs & counts):")
    for val, count in zip(unique_vals, count_vals):
        print(f"  Chord ID {val:02d}: {count} occurrences")
        
    if len(unique_vals) <= 1:
        print("⚠️ Warning: Only one unique label class found in the sanity batch! Verify your dataset labels.")
    else:
        print("✅ Labels display dynamic distribution. Ready to train.")
    print("=========================\n")
        
    # 4. Model initialization
    input_bins = 84
    num_classes = 25
    
    if model_type.lower() == 'transformer':
        model = TransformerChordRecognizer(input_bins=input_bins, num_classes=num_classes)
        print("Initialized TransformerChordRecognizer model.")
    elif model_type.lower() == 'crnn':
        model = CRNNChordBaseline(input_bins=input_bins, num_classes=num_classes)
        print("Initialized CRNNChordBaseline model.")
    else:
        raise ValueError(f"Unrecognized model type: {model_type}")
        
    model = model.to(device)
    
    # 5. Optimizer, Criterion (Focal Loss), and LR Scheduler
    # Note: Reduced learning rate (1e-4) for stable Transformer training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    best_val_loss = float('inf')
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        # --- Training Epoch ---
        model.train()
        running_loss = 0.0
        train_batches = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for X_batch, y_batch in train_batches:
            X_batch = X_batch.to(device)  # shape: [batch_size, 84, num_frames]
            
            # --- AUDIO DSP FIX: Log-Compression & Z-Score Normalization ---
            # Compress huge linear amplitudes to logarithmic scale
            X_batch = torch.log1p(X_batch * 10.0)
            
            # Instance-wise Z-Score normalization (Mean=0, Std=1)
            mean = X_batch.mean(dim=(1, 2), keepdim=True)
            std = X_batch.std(dim=(1, 2), keepdim=True) + 1e-8
            X_batch = (X_batch - mean) / std
            # --------------------------------------------------------------
            
            # Permute CQT from [batch_size, 84, num_frames] to [batch_size, num_frames, 84] for model input
            X_batch = X_batch.permute(0, 2, 1)
            y_batch = y_batch.to(device)  # shape: [batch_size, num_frames]
            
            optimizer.zero_grad()
            
            # Forward pass: outputs shape [batch_size, num_frames, 25]
            logits = model(X_batch)
            
            # CRITICAL: Loss expects input of shape [batch_size, num_classes, num_frames]
            logits_permuted = logits.permute(0, 2, 1)
            loss = criterion(logits_permuted, y_batch)
            
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients in Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            train_batches.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # --- Validation Epoch ---
        model.eval()
        running_val_loss = 0.0
        correct_frames = 0
        total_frames = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                
                # --- AUDIO DSP FIX (Apply same normalization to validation) ---
                X_val = torch.log1p(X_val * 10.0)
                mean = X_val.mean(dim=(1, 2), keepdim=True)
                std = X_val.std(dim=(1, 2), keepdim=True) + 1e-8
                X_val = (X_val - mean) / std
                # --------------------------------------------------------------
                
                # Permute for model input
                X_val = X_val.permute(0, 2, 1)
                y_val = y_val.to(device)
                
                val_logits = model(X_val)
                
                # Permute logits for loss calculation
                val_logits_permuted = val_logits.permute(0, 2, 1)
                val_loss = criterion(val_logits_permuted, y_val)
                running_val_loss += val_loss.item() * X_val.size(0)
                
                # Basic frame-level accuracy
                preds = val_logits.argmax(dim=2)  # shape: [batch_size, num_frames]
                correct_frames += (preds == y_val).sum().item()
                total_frames += y_val.numel()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_frames / total_frames if total_frames > 0 else 0.0
        
        # Print epoch summaries
        print(f"Epoch {epoch}/{epochs} Summary | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Accuracy: {epoch_val_acc * 100:.2f}%")
        
        # Step LR Scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        
        # --- Checkpointing ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_path = "models/best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Saved new best model to {checkpoint_path} with Val Loss: {epoch_val_loss:.4f}")
            
    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    # Defaulting to 50 epochs for a solid training run, lr reduced to 1e-4
    train_model(processed_dir='data/processed', epochs=50, lr=1e-4)