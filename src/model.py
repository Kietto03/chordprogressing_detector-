import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject sequential order into the Transformer.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # division term for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class CRNNChordBaseline(nn.Module):
    """
    CRNN Baseline for Audio Chord Recognition.
    Applies Conv1d layers along the time frames (treating CQT bins as channels)
    followed by a Bidirectional GRU and a Linear projection to 25 chord classes.
    """
    def __init__(self, input_bins=84, num_classes=25, rnn_hidden=128, dropout=0.1):
        super().__init__()
        # Conv1d expects input as [batch_size, in_channels, seq_len]
        # We treat CQT frequency bins (84) as input channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_bins, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU expects input as [batch_size, seq_len, input_size]
        self.rnn = nn.GRU(
            input_size=128, 
            hidden_size=rnn_hidden, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_classes > 1 else 0.0 # PyTorch GRU dropout only applies if num_layers > 1
        )
        
        # Fully connected maps bidirectional GRU output (hidden_size * 2) to chord classes
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_bins] (e.g. [32, 215, 84])
        
        # 1. Permute to [batch_size, input_bins, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # 2. Extract local spatial/spectral features
        x = self.conv(x)  # shape: [batch_size, 128, seq_len]
        
        # 3. Permute back to [batch_size, seq_len, 128] for GRU
        x = x.permute(0, 2, 1)
        
        # 4. Process temporal dependencies
        x, _ = self.rnn(x)  # shape: [batch_size, seq_len, rnn_hidden * 2]
        
        # 5. Classify each frame
        logits = self.fc(x)  # shape: [batch_size, seq_len, num_classes] (e.g. [32, 215, 25])
        return logits


class TransformerChordRecognizer(nn.Module):
    """
    Transformer-based Frame-level Chord Recognizer.
    Projects 84 CQT bins to a higher embedding space, adds Positional Encoding,
    processes with a multi-layer Transformer Encoder, and classifies each frame.
    """
    def __init__(self, input_bins=84, num_classes=25, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Projects [batch_size, seq_len, 84] -> [batch_size, seq_len, d_model]
        self.input_projection = nn.Linear(input_bins, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_bins]
        
        # 1. Projection to embedding dimension
        x = self.input_projection(x)  # shape: [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model) # CRITICAL FIX
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)  # shape: [batch_size, seq_len, d_model]
        
        # 3. Process with Transformer Encoder (Self-Attention across time frames)
        x = self.transformer_encoder(x)  # shape: [batch_size, seq_len, d_model]
        
        # 4. Classify each frame
        logits = self.fc(x)  # shape: [batch_size, seq_len, num_classes]
        return logits


if __name__ == "__main__":
    print("Testing Audio Chord Recognition Models...")
    
    # Define hyperparams matching the dataset chunk specs
    batch_size = 4
    num_frames = 215
    input_bins = 84
    num_classes = 25
    
    # Create dummy tensor representing [batch_size, seq_len, input_bins]
    dummy_input = torch.randn(batch_size, num_frames, input_bins)
    print(f"Dummy Input shape: {dummy_input.shape}")
    
    # 1. Instantiate and run CRNN
    print("\n--- Testing CRNN baseline model ---")
    crnn_model = CRNNChordBaseline(input_bins=input_bins, num_classes=num_classes)
    crnn_output = crnn_model(dummy_input)
    print(f"CRNN output shape: {crnn_output.shape}")
    assert crnn_output.shape == (batch_size, num_frames, num_classes), "CRNN output shape is incorrect!"
    
    # 2. Instantiate and run Transformer
    print("\n--- Testing Transformer model ---")
    transformer_model = TransformerChordRecognizer(input_bins=input_bins, num_classes=num_classes)
    transformer_output = transformer_model(dummy_input)
    print(f"Transformer output shape: {transformer_output.shape}")
    assert transformer_output.shape == (batch_size, num_frames, num_classes), "Transformer output shape is incorrect!"
    
    print("\nVerification successful! All model forward passes completed with correct shapes.")
