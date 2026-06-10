import os
import sys
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# sys.path hack removed in PR10 (rely on `pip install -e .` or PYTHONPATH=. from packaging in pyproject.toml)
# If running directly without install: PYTHONPATH=. python src/train.py ...

from src.config import (
    DEFAULT_SEED,
    MODEL_DIR,
    RUN_CONFIG_PREFIX,
    CHUNK_LENGTH_FRAMES,
    N_BINS,
    NUM_CLASSES,
)
from src.utils import set_all_seeds
from src.dataset import get_dataloaders
from src.model import TransformerChordRecognizer, CRNNChordBaseline
from src.tracking import ExperimentTracker  # PR8 lightweight tracking

# Optional MIREX metrics (PR7)
try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False

from sklearn.metrics import classification_report, accuracy_score
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss helps combat extreme class imbalance by down-weighting the loss 
    assigned to well-classified examples (like C:maj or N) and focusing on hard examples.
    Supports class-specific alpha weighting.
    """
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # alpha can be a float, a 1D tensor, or None
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs shape: [batch_size, num_classes, seq_len]
        # targets shape: [batch_size, seq_len]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Gather alpha weights based on target class IDs
                w = self.alpha[targets] # shape: [batch_size, seq_len]
                focal_loss = w * focal_loss
            else:
                focal_loss = self.alpha * focal_loss
                
        return focal_loss.mean()


def compute_class_weights(dataloader, num_classes=25):
    """
    Compute class weights dynamically based on the inverse frequency of target classes in the training set.
    """
    print("Computing class weights dynamically from training dataset labels...")
    class_counts = np.zeros(num_classes)
    for _, y_batch in dataloader:
        unique, counts = np.unique(y_batch.numpy(), return_counts=True)
        for u, c in zip(unique, counts):
            if 0 <= u < num_classes:
                class_counts[u] += c
                
    # Safeguard against zero count classes
    class_counts = np.clip(class_counts, a_min=1, a_max=None)
    
    # Calculate inverse frequency
    total_frames = class_counts.sum()
    class_weights = total_frames / (num_classes * class_counts)
    
    # Normalize weights so that their mean is 1.0
    class_weights = class_weights / class_weights.mean()
    return torch.FloatTensor(class_weights)


def train_model(
    processed_dir: str = 'data/processed',
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_type: str = 'transformer',
    loss_type: str = 'focal',
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    metrics_level: str = 'basic',
):
    """
    Train either CRNN or Transformer model on CQT frame features.

    seed + deterministic: full reproducibility support (PR5).
    """
    # PR5: Reproducibility first
    set_all_seeds(seed, deterministic=deterministic)
    print(f"Using seed={seed} (deterministic={deterministic})")

    # PR8: Lightweight tracking
    tracker = ExperimentTracker(log_dir="runs", use_tensorboard=True)
    tracker.log_hparams({
        "seed": seed,
        "model_type": model_type,
        "lr": lr,
        "batch_size": batch_size,
    })

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
        
    # 2. Get DataLoaders (PR5: pass seed for reproducible song-level split)
    try:
        train_loader, val_loader = get_dataloaders(
            processed_dir, batch_size=batch_size, train_split=0.8, seed=seed
        )
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
    
    # 5. Compute class weights and setup Criterion
    class_weights = compute_class_weights(train_loader, num_classes=num_classes).to(device)
    print(f"Dynamic class weights calculated (mean=1.0):\n{class_weights.cpu().numpy()}")
    
    if loss_type.lower() == 'focal':
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
        print("Using dynamic weighted Focal Loss.")
    elif loss_type.lower() == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using dynamic weighted Cross Entropy Loss.")
    else:
        raise ValueError(f"Unrecognized loss type: {loss_type}")
        
    # 6. Optimizer and LR Scheduler
    # Note: Reduced learning rate (1e-4) for stable Transformer training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        
        all_preds = []
        all_targets = []
        
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
                
                # Collect for per-class / MIREX metrics (PR7)
                all_preds.append(preds.cpu().numpy().flatten())
                all_targets.append(y_val.cpu().numpy().flatten())
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_frames / total_frames if total_frames > 0 else 0.0
        
        # PR7: MIREX-style + per-class metrics
        flat_preds = np.concatenate(all_preds) if all_preds else np.array([])
        flat_targets = np.concatenate(all_targets) if all_targets else np.array([])
        
        per_class_f1 = {}
        non_n_acc = 0.0
        symbol_overlap = 0.0
        if len(flat_targets) > 0:
            # sklearn report (per-class)
            try:
                report = classification_report(flat_targets, flat_preds, labels=list(range(25)), output_dict=True, zero_division=0)
                per_class_f1 = {f"class_{k}": v['f1-score'] for k, v in report.items() if k.isdigit()}
                print("Per-class F1 (sample):", {k: round(v, 3) for k, v in list(per_class_f1.items())[:5]})
            except Exception:
                pass
            
            # Non-N accuracy
            non_n_mask = flat_targets != 24
            if non_n_mask.sum() > 0:
                non_n_acc = (flat_preds[non_n_mask] == flat_targets[non_n_mask]).mean()
            
            # Basic chord symbol overlap (root + quality match under our 25-class vocab)
            # Treat as exact match on class id (simplified "symbol recall")
            symbol_overlap = (flat_preds == flat_targets).mean()
        
        # Print epoch summaries (enhanced with PR7 metrics)
        print(f"Epoch {epoch}/{epochs} Summary | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Accuracy: {epoch_val_acc * 100:.2f}% | "
              f"Non-N Acc: {non_n_acc*100:.2f}% | "
              f"Symbol Overlap: {symbol_overlap*100:.2f}%")
        
        # PR8: Log to tracker / TB
        tracker.log_scalar("train/loss", epoch_train_loss, epoch)
        tracker.log_scalar("val/loss", epoch_val_loss, epoch)
        tracker.log_scalar("val/acc", epoch_val_acc, epoch)
        if 'non_n_acc' in locals():
            tracker.log_scalar("val/non_n_acc", non_n_acc, epoch)
        if 'symbol_overlap' in locals():
            tracker.log_scalar("val/symbol_overlap", symbol_overlap, epoch)
        
        # Step LR Scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        
        # --- Checkpointing (PR5: timestamped + keep best_*.pth for backward compat) ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            suffix = "_crnn" if model_type.lower() == 'crnn' else ""
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(MODEL_DIR, exist_ok=True)
            checkpoint_path = os.path.join(MODEL_DIR, f"{model_type}_best_{ts}{suffix}.pth")
            torch.save(model.state_dict(), checkpoint_path)

            # Maintain the "latest" pointer the rest of the app/notebook expect
            best_pointer = os.path.join(MODEL_DIR, f"best_model{suffix}.pth")
            try:
                import shutil
                shutil.copy2(checkpoint_path, best_pointer)
            except Exception:
                pass

            print(f"--> Saved new best model to {checkpoint_path} with Val Loss: {epoch_val_loss:.4f}")

    # PR5/PR7: Write run metadata for reproducibility + MIREX-style metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "deterministic": deterministic,
        "model_type": model_type,
        "loss_type": loss_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_val_loss": best_val_loss,
        "processed_dir": processed_dir,
        "chunk_length_frames": CHUNK_LENGTH_FRAMES,
        "n_bins": N_BINS,
        "num_classes": NUM_CLASSES,
        "final_val_acc": float(epoch_val_acc) if 'epoch_val_acc' in locals() else None,
        "final_non_n_acc": float(non_n_acc) if 'non_n_acc' in locals() else None,
        "final_symbol_overlap": float(symbol_overlap) if 'symbol_overlap' in locals() else None,
        "per_class_f1_sample": {k: float(v) for k, v in list(per_class_f1.items())[:5]} if 'per_class_f1' in locals() and per_class_f1 else {},
        "metrics_level": metrics_level,
        "mir_eval_available": HAS_MIR_EVAL,
    }
    if metrics_level == 'mir' and HAS_MIR_EVAL and len(flat_targets) > 0:
        try:
            # Simple mir_eval usage for chord (root/quality simplified)
            # For our 25-class, we can map back but keep lightweight
            run_meta["mir_eval_note"] = "mir_eval available - extend with mir_eval.chord for full CSR/WAOR if desired"
        except Exception:
            pass

    run_file = os.path.join(MODEL_DIR, f"{RUN_CONFIG_PREFIX}_{model_type}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(run_file, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"Run metadata + MIREX-style metrics written to: {run_file}")

    tracker.close()  # PR8

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Audio Chord Recognition Models")
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Path to processed h5 files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'crnn'], help='Model architecture')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'ce'], help='Loss function')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for full reproducibility (PR5)')
    parser.add_argument('--no_deterministic', action='store_true', help='Disable cuDNN deterministic mode (faster on GPU but less reproducible)')
    parser.add_argument('--metrics', type=str, default='basic', choices=['basic', 'mir'], help='Metrics level (PR7): basic (sklearn + overlap) or mir (try mir_eval)')
    
    args = parser.parse_args()
    
    train_model(
        processed_dir=args.processed_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        loss_type=args.loss_type,
        seed=args.seed,
        deterministic=not args.no_deterministic,
        metrics_level=args.metrics,
    )