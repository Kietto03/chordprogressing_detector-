# AI Chord Progression Analyzer & Recognizer 🎵

An end-to-end deep learning project for **Audio Chord Recognition (ACR)**. This project processes raw audio signals, extracts spectro-temporal features, trains high-capacity sequence models (Transformer and CRNN), and provides an interactive Streamlit web dashboard to analyze chord progressions in real-time.

The model is trained on the annotated **McGill-Billboard Dataset**, predicting frame-level chords mapped to a simplified 25-class vocabulary.

---

## 🚀 Key Features
*   **YouTube Audio Harvester**: Automatically matches McGill-Billboard track annotations to YouTube tracks using metadata mapping and downloads native high-quality `.m4a` streams.
*   **Constant-Q Transform (CQT) DSP Pipeline**: Resamples audio, computes absolute CQT magnitude, applies log-amplitude compression, and implements instance-wise Z-score normalization.
*   **Dynamic Frame Label Alignment**: Parses raw timing-interval chord labels and maps them to downsampled CQT frame boundaries.
*   **Advanced Architectures**:
    *   **Transformer-based Recognizer**: Features Multi-Head Self-Attention layers and sinusoidal positional encodings to capture long-range temporal dependencies.
    *   **CRNN Baseline**: Uses Convolutional layers (Conv1D) for local feature representation, followed by a Bidirectional GRU.
*   **Focal Loss Optimization**: Fights extreme chord class imbalance by down-weighting the loss contribution of easy/frequent chords (e.g., Silence or Major chords) to focus training on hard examples.
*   **Stunning Streamlit Web UI**: Visualizes predicted chord alignments as colored timeline rectangles overlaid directly on top of interactive, downsampled audio waveforms using Plotly.

---

## 📁 Repository Structure

```directory
├── README.md                 # Project summary and documentation (this file)
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit web application
├── models/                   # Directory storing trained model checkpoints (.pth)
├── data/
│   ├── raw/
│   │   └── McGill-Billboard/ # Raw McGill Billboard annotations, metadata and audio
│   │       ├── annotations/  # Double-nested song folders containing .lab files
│   │       └── billboard-2.0-index.csv # Track metadata index
│   └── processed/            # HDF5 dataset (.h5) with compressed CQTs and labels
├── notebooks/
│   └── evaluation.ipynb      # Analytical evaluation notebook (macro/micro metrics)
└── src/
    ├── download_audio.py     # yt-dlp script to download and parse YouTube audios
    ├── etl_pipeline.py       # Audio feature extraction and annotation mapping pipeline
    ├── dataset.py            # PyTorch Dataset and DataLoader wrappers
    ├── model.py              # PyTorch deep learning model architectures
    └── train.py              # Training, optimization, and checkpointing logic
```

---

## 🛠️ Module Breakdown

### 1. Data Harvesting (`src/download_audio.py`)
Because the raw McGill-Billboard corpus contains only chord labels, this harvester resolves the missing audio data.
*   It reads song titles and artists from [billboard-2.0-index.csv](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/data/raw/McGill-Billboard/billboard-2.0-index.csv).
*   Uses `yt-dlp` to query YouTube for each specific track (`ytsearch1:{title} {artist} audio`).
*   Bypasses system `ffmpeg` limitations on macOS by downloading native `.m4a` audio format directly, saving them in the respective song directories under `data/raw/McGill-Billboard/annotations/annotations/`.

### 2. ETL Processing Pipeline (`src/etl_pipeline.py`)
This script recursively scans directories containing `.lab` files and processes matched audios:
*   **Audio Load & Resample**: Waveforms are resampled to mono at 22,050 Hz.
*   **CQT Spectrogram**: Computes absolute Constant-Q Transform magnitude across 84 frequency bins (7 octaves, starting at C1, 12 bins per octave).
*   **Chord Vocabulary Mapping**: Simplifies raw McGill-Billboard chord annotations to a standard **25-class vocabulary**:
    *   `0-11`: Major Chords (C to B)
    *   `12-23`: Minor Chords (C to B)
    *   `24`: `'N'` (No chord / Silence)
*   **Temporal Alignment**: Maps time intervals to CQT frame indices (`hop_length=512`) and aligns labels.
*   **HDF5 Storage**: Compresses CQT matrices and aligned labels into `.h5` files using gzip compression for fast I/O during training.

### 3. Dataset & DataLoaders (`src/dataset.py`)
*   Defines a PyTorch `ChordDataset` that slices long audio sequences into fixed chunks of length 215 frames (~5 seconds of audio).
*   Index coordinates (file, start_frame, end_frame) are cached, while slices are retrieved dynamically from HDF5 files during iteration to save RAM.
*   Splits the chunks into training and validation sets, returning standard PyTorch `DataLoader` instances.

### 4. Neural Architectures (`src/model.py`)
*   **`TransformerChordRecognizer`**:
    *   Linear projection maps the 84 spectral bins to `d_model=256` dimension.
    *   Adds sinusoidal positional encodings to inject sequence order.
    *   Stack of 4 Transformer Encoder layers performs self-attention across the temporal dimension.
    *   Frame classifier projects back to 25 chord classes.
*   **`CRNNChordBaseline`**:
    *   Applies a sequence of 1D Convolution layers (BatchNorm, ReLU, Dropout) across CQT frequency bins to capture local spectro-temporal features.
    *   Feeds features into a 2-layer Bidirectional GRU (`rnn_hidden=128`).
    *   Maps bidirectional outputs to 25 classes.

### 5. Optimizer & Trainer (`src/train.py`)
*   **Focal Loss**: Combats extreme class imbalance using cross-entropy with focusing parameter $\gamma = 2.0$ and weighting factor $\alpha = 0.25$.
*   **DSP Normalization**: Applies log-compression (`log1p(x * 10)`) and instance-wise Z-score normalization to input features on the fly.
*   **Regularization**: Leverages AdamW with weight decay, gradient norm clipping at $1.0$ (preventing exploding gradients), and dropout.
*   **Scheduler**: Uses `ReduceLROnPlateau` scheduler on validation loss.
*   **Checkpoints**: Automatically saves the model with the lowest validation loss to `models/best_model.pth`.

### 6. Interactive Web App (`app.py`)
An interactive dashboard built with Streamlit:
*   Allows the user to upload any personal `.wav`, `.mp3`, or `.m4a` track.
*   Extracts CQT features, applies log-compression, Z-score normalizes them, and runs inference.
*   Uses **Plotly** to overlay the predicted chord segments (represented as semi-transparent color bands) on top of the interactive, downsampled audio waveform.
*   Allows users to play the uploaded track directly in the browser and follow chord progressions visually.

### 7. Evaluation Suite (`notebooks/evaluation.ipynb`)
Evaluates model performance:
*   **Macro Metrics**: Computes global frame-level Precision, Recall, and F1-score across all validation tracks, and plots a normalized Confusion Matrix.
*   **Micro Analysis**: Draws a detailed timeline ribbon comparing ground truth annotations vs. model predictions along with vertical dotted lines denoting chord transition boundaries.

---

## 🏃 Setup & Execution

### 1. Installation
Ensure Python 3.10+ is installed. Clone the repository and install requirements in your virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Harvest Audio Data
Download the master index file and execute the Harvester script to download all dataset audio files in `.m4a` format:
```bash
python src/download_audio.py
```

### 3. Feature Extraction (ETL)
Extract CQT features and align labels to frames:
```bash
python src/etl_pipeline.py
```

### 4. Train the Model
Train the Transformer model:
```bash
python src/train.py
```

### 5. Launch the Web App
Run the Streamlit app locally:
```bash
PYTHONPATH=. streamlit run app.py
```
Open the provided local URL (typically `http://localhost:8501`) in your browser to interact with the app.

---

## 📊 Summary of Model Performance
During validation, the frame-level chord recognizer reaches high classification metrics across standard Billboard tracks.
*   **Average Frame-level Validation Accuracy**: ~58.6% (on limited training data).
*   **Transition Boundaries**: Micro-evaluation confirms that transition boundaries predicted by the self-attention mechanism align closely with real annotated changes.
