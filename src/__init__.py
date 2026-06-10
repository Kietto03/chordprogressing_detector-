"""
AI Chord Progression Analyzer & Recognizer

End-to-end deep learning system for Audio Chord Recognition (ACR)
using Constant-Q Transform features and Transformer/CRNN models
trained on the McGill-Billboard dataset.

See docs/implementation-plan.md for the full modernization roadmap.
"""

__version__ = "0.1.0"

# Main public exports (convenience)
from .model import TransformerChordRecognizer, CRNNChordBaseline
from .etl_pipeline import ChordVocabularyMapper, AudioETLPipeline
from .dataset import ChordDataset, get_dataloaders
from .train import train_model, FocalLoss
from .utils import set_all_seeds
from .config import *  # constants for reproducibility

__all__ = [
    "__version__",
    "TransformerChordRecognizer",
    "CRNNChordBaseline",
    "ChordVocabularyMapper",
    "AudioETLPipeline",
    "ChordDataset",
    "get_dataloaders",
    "train_model",
    "FocalLoss",
    "set_all_seeds",
]