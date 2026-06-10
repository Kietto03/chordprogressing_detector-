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
    Optimized CRNN Baseline for Audio Chord Recognition.
    Applies Conv2d layers along the spectro-temporal dimensions (time and frequency)
    followed by a Bidirectional GRU and a Linear projection to 25 chord classes.
    """
    def __init__(self, input_bins=84, num_classes=25, rnn_hidden=128, dropout=0.1):
        super().__init__()
        # Conv2D expects input as [batch_size, 1, seq_len, input_bins]
        # We downsample the frequency axis while preserving temporal frame rate
        self.conv_frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Downsamples freq: 84 -> 42
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Downsamples freq: 42 -> 21
        )
        
        self.flat_dim = 64 * (input_bins // 4)
        
        # Linear projection to map Conv2D flat features to GRU input size
        self.input_projection = nn.Linear(self.flat_dim, 128)
        self.input_dropout = nn.Dropout(dropout)
        
        # GRU expects input as [batch_size, seq_len, input_size]
        self.rnn = nn.GRU(
            input_size=128, 
            hidden_size=rnn_hidden, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout
        )
        
        # LayerNorm stabilizes activations prior to classification
        self.norm = nn.LayerNorm(rnn_hidden * 2)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_bins] (e.g. [32, 215, 84])
        
        # 1. Reshape to [batch_size, 1, seq_len, input_bins] for Conv2D
        x_conv = x.unsqueeze(1)
        
        # 2. Extract spectro-temporal features
        feat = self.conv_frontend(x_conv)  # shape: [batch_size, 64, seq_len, 21]
        
        # 3. Permute and flatten to [batch_size, seq_len, flat_dim]
        feat = feat.permute(0, 2, 1, 3).contiguous()
        feat = feat.view(feat.size(0), feat.size(1), -1)
        
        # 4. Map to GRU dimension
        x = self.input_projection(feat)
        x = self.input_dropout(x)
        
        # 5. Process temporal dependencies
        x, _ = self.rnn(x)  # shape: [batch_size, seq_len, rnn_hidden * 2]
        
        # 6. Apply normalization
        x = self.norm(x)
        
        # 7. Classify each frame
        logits = self.fc(x)  # shape: [batch_size, seq_len, num_classes]
        return logits


class TransformerChordRecognizer(nn.Module):
    """
    Optimized Transformer-based Frame-level Chord Recognizer.
    Projects 2D-Convolutional features to embedding space, adds Positional Encoding,
    processes with a multi-layer Transformer Encoder, and classifies each frame.
    """
    def __init__(self, input_bins=84, num_classes=25, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Conv2D Front-End preserves temporal framerate, downsamples frequency by 4
        self.conv_frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Downsamples freq: 84 -> 42
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Downsamples freq: 42 -> 21
        )
        
        self.flat_dim = 64 * (input_bins // 4)
        self.d_model = d_model
        
        # Linear projection maps flattened features to d_model
        self.input_projection = nn.Linear(self.flat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Transformer Encoder Stack
        # enable_nested_tensor=False to suppress the Pre-LN warning
        # (see https://github.com/pytorch/pytorch/issues/100988 and design plan)
        # Try to pass enable_nested_tensor=False to suppress the Pre-LN warning on older PyTorch versions.
        # Fall back if the argument has been removed in newer PyTorch versions.
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                enable_nested_tensor=False,
            )
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CRITICAL FIX: Add LayerNorm at output of Pre-LN Transformer encoder
        self.final_norm = nn.LayerNorm(d_model)
        
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_bins]
        
        # 1. Reshape to [batch_size, 1, seq_len, input_bins] for Conv2D
        x_conv = x.unsqueeze(1)
        
        # 2. Extract spectro-temporal features
        feat = self.conv_frontend(x_conv)  # shape: [batch_size, 64, seq_len, 21]
        
        # 3. Permute and flatten to [batch_size, seq_len, flat_dim]
        feat = feat.permute(0, 2, 1, 3).contiguous()
        feat = feat.view(feat.size(0), feat.size(1), -1)
        
        # 4. Projection to embedding dimension
        x = self.input_projection(feat)  # shape: [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model)
        
        # 5. Add Positional Encoding and Dropout
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        
        # 6. Process with Transformer Encoder (Self-Attention across time frames)
        x = self.transformer_encoder(x)  # shape: [batch_size, seq_len, d_model]
        
        # 7. Apply final normalization
        x = self.final_norm(x)
        
        # 8. Classify each frame
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
