import os
import re
import logging
import numpy as np
import h5py
import pandas as pd
import librosa
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChordVocabularyMapper:
    """
    Maps raw McGill Billboard chord notations to a simplified 25-class vocabulary:
    0-11: Major chords (C to B)
    12-23: Minor chords (C to B)
    24: 'N' (No chord/Silence)
    """
    def __init__(self):
        # 12 pitch classes starting from C
        self.roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.root_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
            'A#': 10, 'Bb': 10, 'B': 11
        }
        # Regex to parse the root note and optional quality/extensions before a slash
        # Example matches: 
        # "C:maj7" -> root="C", quality="maj7"
        # "G:min(*5)" -> root="G", quality="min(*5)"
        # "A:sus4/b7" -> root="A", quality="sus4"
        # "Db" -> root="Db", quality=None
        self.chord_regex = re.compile(r'^([A-G][#b]?)(?::([^/]+))?')

    def map_chord(self, raw_chord_string):
        """
        Parse raw chord string using regex and return the 25-class integer index.
        """
        if not raw_chord_string or raw_chord_string.strip() in ['N', 'N/A', '']:
            return 24
            
        raw_chord_string = raw_chord_string.strip()
        
        match = self.chord_regex.match(raw_chord_string)
        if not match:
            # Fallback complex/unrecognized format to No Chord
            return 24
            
        root = match.group(1)
        quality_str = match.group(2)
        
        root_idx = self.root_map.get(root)
        if root_idx is None:
            # Fallback if the parsed root note is unrecognized
            return 24
            
        # Determine quality: default to major, check for minor patterns
        is_minor = False
        if quality_str:
            quality_str = quality_str.lower()
            # If the quality string contains minor/diminished keywords:
            # 'min' (minor), 'dim' (diminished), 'hdim' (half-diminished), 
            # or starts with 'm' (but not 'maj' or 'max')
            if ('min' in quality_str or 
                'dim' in quality_str or 
                'hdim' in quality_str or 
                (quality_str.startswith('m') and not quality_str.startswith('maj'))):
                is_minor = True
                
        if is_minor:
            return 12 + root_idx
        else:
            return root_idx


class AudioETLPipeline:
    def __init__(self, raw_dir='data/raw/McGill-Billboard', processed_dir='data/processed', target_sr=22050, hop_length=512, n_bins=84):
        """
        Initialize the ETL pipeline with directories and audio processing parameters.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.target_sr = target_sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        self.chord_mapper = ChordVocabularyMapper()
        logger.info(f"Initialized AudioETLPipeline with raw_dir: {self.raw_dir}, processed_dir: {self.processed_dir}")

    def extract_audio(self, file_path):
        """
        Load audio file as mono and resample to target_sr.
        """
        try:
            logger.info(f"Extracting and resampling audio from: {file_path}")
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return y
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None

    def transform_to_cqt(self, y):
        """
        Convert waveform to Constant-Q Transform (CQT) spectrogram.
        Returns absolute magnitude.
        """
        try:
            # C1 is note C1 (~32.70 Hz)
            fmin = librosa.note_to_hz('C1')
            C = librosa.cqt(
                y, 
                sr=self.target_sr, 
                hop_length=self.hop_length, 
                fmin=fmin, 
                n_bins=self.n_bins, 
                bins_per_octave=12
            )
            return np.abs(C)
        except Exception as e:
            logger.error(f"Error computing CQT: {e}")
            return None

    def load_to_hdf5(self, song_id, cqt_matrix, labels_array):
        """
        Save the CQT matrix and aligned labels to HDF5 file with gzip compression.
        """
        try:
            output_path = os.path.join(self.processed_dir, f"{song_id}.h5")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('cqt', data=cqt_matrix, compression='gzip')
                f.create_dataset('labels', data=labels_array, compression='gzip')
            logger.info(f"Successfully saved CQT matrix and labels to HDF5: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving HDF5 for song {song_id}: {e}")
            return None

    def _generate_dummy_audio(self, duration=30):
        """
        Generate a dummy sine wave audio waveform of specified duration (in seconds).
        """
        logger.info(f"Generating dummy audio of duration {duration:.2f}s at sample rate {self.target_sr}")
        # Standard A440 sine wave
        t = np.linspace(0, duration, int(self.target_sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        return y

    def parse_labels(self, label_path):
        """
        Read the .lab file using pandas. Format is: start_time end_time chord_name
        """
        try:
            df = pd.read_csv(label_path, sep=r'\s+', header=None, names=['start_time', 'end_time', 'chord'])
            return df
        except Exception as e:
            logger.error(f"Error parsing lab file {label_path}: {e}")
            return None

    def align_labels_to_frames(self, labels_df, num_frames):
        """
        Generate a 1D numpy array of length num_frames filled with mapped chord IDs.
        """
        # Default initialization with No Chord 'N' (index 24)
        frame_labels = np.full(num_frames, 24, dtype=np.int32)
        
        for _, row in labels_df.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            chord = row['chord']
            
            chord_id = self.chord_mapper.map_chord(chord)
            
            # Map time bounds to frame indices
            start_frame = int(librosa.time_to_frames(start_time, sr=self.target_sr, hop_length=self.hop_length))
            end_frame = int(librosa.time_to_frames(end_time, sr=self.target_sr, hop_length=self.hop_length))
            
            # Bound and clip indices
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))
            
            if start_frame < end_frame:
                frame_labels[start_frame:end_frame] = chord_id
                
        return frame_labels

    def run_pipeline(self):
        """
        Iterate through raw folders, locate label files, find/generate audio,
        transform to CQT, align labels to frames, and save as HDF5 datasets.
        Process the first 5 folders (as a test run).
        """
        logger.info("Starting ETL pipeline...")
        if not os.path.exists(self.raw_dir):
            logger.error(f"Raw directory does not exist: {self.raw_dir}")
            return

        audio_extensions = ('.mp3', '.wav', '.ogg', '.flac', '.m4a')
        processed_count = 0
        target_limit = 5

        # Scan raw directory recursively for folders containing .lab files
        folders_to_check = []
        for root, dirs, files in os.walk(self.raw_dir):
            lab_files = [f for f in files if f.lower().endswith('.lab')]
            if lab_files:
                folders_to_check.append((root, lab_files, files))

        logger.info(f"Found {len(folders_to_check)} folders containing .lab files.")
        if not folders_to_check:
            logger.warning("No folders with labels found. Pipeline run skipped.")
            return

        # Sort the folders for consistent ordering
        folders_to_check.sort(key=lambda x: x[0])
        folders_to_process = folders_to_check[:target_limit]

        for root, lab_files, all_files in tqdm(folders_to_process, desc="Processing Song Folders"):
            # Select preferred annotation file if multiple exist
            lab_file = None
            for preferred in ['majmin.lab', 'majmin7.lab', 'majmininv.lab', 'majmin7inv.lab']:
                if preferred in lab_files:
                    lab_file = preferred
                    break
            if not lab_file:
                lab_file = lab_files[0]

            lab_path = os.path.join(root, lab_file)
            labels_df = self.parse_labels(lab_path)
            if labels_df is None or labels_df.empty:
                logger.warning(f"Skipping {root} due to empty or unparseable labels file.")
                continue

            # Look for audio file
            audio_files = [f for f in all_files if f.lower().endswith(audio_extensions)]
            y = None

            if audio_files:
                audio_file = audio_files[0]
                audio_path = os.path.join(root, audio_file)
                y = self.extract_audio(audio_path)

            if y is None:
                # If no audio, print warning and generate dummy sine wave audio matching total duration
                total_duration = labels_df.iloc[-1]['end_time']
                if total_duration <= 0:
                    total_duration = 30.0  # Fallback duration
                logger.warning(f"No audio file in {root}. Using generated dummy audio of {total_duration:.2f}s.")
                y = self._generate_dummy_audio(total_duration)

            # CQT extraction
            cqt_matrix = self.transform_to_cqt(y)
            if cqt_matrix is None:
                logger.error(f"Failed to extract CQT for folder {root}. Skipping.")
                continue

            # Extract dimensions
            num_frames = cqt_matrix.shape[1]

            # Frame label alignment
            labels_array = self.align_labels_to_frames(labels_df, num_frames)

            # Define song ID based on directory name (e.g. "1069")
            dir_name = os.path.basename(root)
            song_id = dir_name if dir_name.isdigit() else os.path.splitext(lab_file)[0]

            # Write HDF5 file
            self.load_to_hdf5(song_id, cqt_matrix, labels_array)
            processed_count += 1

        logger.info(f"Pipeline finished. Processed {processed_count} files.")


if __name__ == "__main__":
    pipeline = AudioETLPipeline()
    pipeline.run_pipeline()
