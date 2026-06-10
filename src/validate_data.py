"""
Data validation script (PR6).

Usage examples:
    python -m src.validate_data --processed_dir data/processed
    python -m src.validate_data --raw_dir data/raw/McGill-Billboard --processed_dir data/processed --strict
"""

import os
import argparse
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm


def validate_processed(processed_dir: str, strict: bool = False):
    """Validate all .h5 files in the processed directory."""
    if not os.path.exists(processed_dir):
        print(f"ERROR: Processed directory not found: {processed_dir}")
        return False

    h5_files = sorted([f for f in os.listdir(processed_dir) if f.lower().endswith('.h5')])
    if not h5_files:
        print(f"No .h5 files found in {processed_dir}")
        return False

    print(f"Validating {len(h5_files)} processed files in {processed_dir}...")
    problems = 0

    for fname in tqdm(h5_files, desc="Validating HDF5"):
        path = os.path.join(processed_dir, fname)
        try:
            with h5py.File(path, 'r') as f:
                if 'cqt' not in f or 'labels' not in f:
                    print(f"  MISSING DATASETS: {fname}")
                    problems += 1
                    continue

                cqt = f['cqt']
                labels = f['labels']

                # Basic shape checks
                if cqt.ndim != 2 or labels.ndim != 1:
                    print(f"  BAD SHAPE: {fname} cqt={cqt.shape} labels={labels.shape}")
                    problems += 1

                if cqt.shape[1] != labels.shape[0]:
                    print(f"  LENGTH MISMATCH: {fname} cqt_frames={cqt.shape[1]} labels={labels.shape[0]}")
                    problems += 1

                # Label range
                lab_min, lab_max = int(labels[:].min()), int(labels[:].max())
                if lab_min < 0 or lab_max > 24:
                    print(f"  BAD LABEL RANGE: {fname} min={lab_min} max={lab_max} (expected 0-24)")
                    problems += 1

                # NaN / Inf in cqt (labels are int, can't be nan)
                cqt_data = cqt[:]
                if np.any(np.isnan(cqt_data)) or np.any(np.isinf(cqt_data)):
                    print(f"  NaN/Inf in CQT: {fname}")
                    problems += 1

                # Optional duration metadata (written by etl_pipeline in PR6)
                if 'audio_duration' in f.attrs and 'lab_duration' in f.attrs:
                    ad = float(f.attrs['audio_duration'])
                    ld = float(f.attrs['lab_duration'])
                    delta = abs(ad - ld)
                    if delta > 2.0:
                        level = "ERROR" if strict else "WARN"
                        print(f"  {level} DURATION MISMATCH: {fname} audio={ad:.2f}s lab={ld:.2f}s delta={delta:.2f}s")

        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
            problems += 1

    print(f"\nValidation complete. Problems found: {problems}")
    return problems == 0


def main():
    parser = argparse.ArgumentParser(description="Validate processed dataset and (optionally) raw audio/lab consistency.")
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Path to processed .h5 files')
    parser.add_argument('--raw_dir', type=str, default=None, help='Optional raw McGill dir for future deeper checks')
    parser.add_argument('--strict', action='store_true', help='Treat duration mismatches as errors')
    args = parser.parse_args()

    ok = validate_processed(args.processed_dir, strict=args.strict)
    if not ok:
        exit(1)
    print("All checks passed (or only warnings in non-strict mode).")


if __name__ == "__main__":
    main()
