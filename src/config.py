"""
Centralized configuration for the Chord Processing Identifier project.

This module defines all magic numbers and hyperparameters in one place
to improve reproducibility, maintainability, and testability (PR5).

All modules should import from here instead of hardcoding values.
"""

# Audio / DSP parameters (used across etl, dataset, train, app, model)
SR: int = 22050                     # Target sample rate
HOP_LENGTH: int = 512               # STFT / CQT hop length
N_BINS: int = 84                    # CQT frequency bins (7 octaves * 12)
FMIN: str = 'C1'                    # Lowest frequency for CQT
CHUNK_LENGTH_FRAMES: int = 215      # ~5 seconds of audio per training chunk
HOP_SIZE: int = 107                 # 50% overlap for sliding-window inference
NUM_CLASSES: int = 25               # 12 major + 12 minor + N (no-chord)

# Training / model hyperparameters
D_MODEL: int = 256
NHEAD: int = 8
NUM_LAYERS: int = 4
DIM_FEEDFORWARD: int = 512
DROPOUT: float = 0.1
RNN_HIDDEN: int = 128               # for CRNN baseline
FOCAL_GAMMA: float = 2.0
FOCAL_ALPHA: float = 0.25

# Data pipeline (PR6)
DURATION_TOLERANCE_SEC: float = 2.0  # Max allowed |audio_dur - lab_dur| before warning/strict error
STRICT_DURATION_CHECK: bool = False  # Default to warning-only for research flexibility

# Reproducibility defaults
DEFAULT_SEED: int = 42

# Misc
MODEL_DIR: str = "models"
RUN_CONFIG_PREFIX: str = "run_config"
