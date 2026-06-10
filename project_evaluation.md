# Professional Project Manager Evaluation & Risk Assessment: AI Chord Progression Analyzer

This document provides a strict, experienced project manager's evaluation of the **AI Chord Progression Analyzer & Recognizer** project. Following a major modernization and engineering push, a comprehensive code audit shows that the critical gaps in security, test automation, software quality assurance, deployment stability, and data ingestion have been successfully resolved. The project is now fully production-ready.

---

## 📊 Executive Summary & Project Rating

### Overall Score: **9.1 / 10** (Status: **PRODUCTION READY / STABLE**)

| Evaluation Dimension | Score | Rating | Verdict & Improvements |
| :--- | :---: | :---: | :--- |
| **Software Quality Assurance (QA)** | 9.5 / 10 | ✅ Passed | Active test suite with `pytest`, coverage reports, and a fully automated GitHub Actions CI pipeline. |
| **Security & Architecture** | 9.5 / 10 | ✅ Passed | Safe, server-driven Plotly timeline rendering. All sandbox-escaping `window.parent` DOM hacks removed. |
| **DevOps & Release Engineering** | 9.0 / 10 | ✅ Passed | version-constrained dependencies in `requirements.txt`, packaging configuration in `pyproject.toml`, and gitignore cleanup complete. |
| **Data Engineering & Ingestion** | 8.5 / 10 | ✅ Passed | Added strict/warn duration mismatch checks in ETL and created a data validation tool (`validate_data.py`). |
| **Data Science & ML Ops** | 9.0 / 10 | ✅ Passed | Centralized config parameters, random seed enforcement, TensorBoard logging, and timestamped best checkpoints. |
| **UI & Usability** | 9.0 / 10 | ✅ Passed | Fully secure, interactive scrub-to-preview Plotly chart and pure Streamlit beat navigation, works in sandboxes. |

#### PM Verdict
The project has successfully transition from a fragile research prototype into a **professional, reproducible, testable, and deployable** software codebase. By eliminating same-origin security violations in the Streamlit frontend, adding a comprehensive automated testing suite, pinning dependencies, and standardizing ML training with random seeds and metrics, the deployment and release blockers have been completely lifted.

---

## 🔍 Audit & Verification of Resolved Issues

### 1. Automated Testing & Verification (Resolved Blocker)
* **Status**: **FULLY RESOLVED**
* **Finding**: A comprehensive test suite has been established under the [tests/](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests) directory, checking all key business logic:
  * [test_etl_vocabulary.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests/test_etl_vocabulary.py): Verifies chord vocabulary mapping and covers edge cases/fallbacks (e.g., no-colon regressions).
  * [test_dataset.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests/test_dataset.py): Checks chunking logic and deterministic song-level train/validation split disjointness.
  * [test_models.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests/test_models.py): Ensures forward passes run with correct shapes and verifies the suppression of PyTorch Pre-LN nested tensor warnings.
  * [test_etl_alignment.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests/test_etl_alignment.py): Tests time-to-frame mapping and boundary clipping.
  * [test_train_smoke.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests/test_train_smoke.py): Verifies Focal Loss forward pass calculations, class weights computation, and train loop checkpointing.
  * [test_app_smokes.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/tests/test_app_smokes.py): Smoke tests app helper functions (MIDI, scale, key, Roman numerals) and asserts norm math equivalence between the training and inference pipelines.
* **CI Integration**: A GitHub Actions workflow [ci.yml](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/.github/workflows/ci.yml) has been set up to automatically run Ruff checks, Mypy type validation, pytest coverage runs, and training/app smoke tests on every push/PR.

### 2. Sandbox Escape & Same-Origin Violation in Streamlit Frontend (Resolved Blocker)
* **Status**: **FULLY RESOLVED**
* **Finding**: The security-violating `window.parent.document` calls inside custom HTML iframe blocks have been completely removed.
* **Refactoring**: 
  * The frontend [app.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/app.py) was rewritten to use standard, sandboxed iframe elements and native Streamlit inputs.
  * A server-side Plotly timeline helper [build_chord_timeline_figure](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/src/viz.py) handles downsampled waveform graphing and overlaps segmented chords as colored rectangles.
  * An interactive `st.slider` acts as a manual scrub-to-preview mechanism, updating the playhead position dynamically in the Plotly chart and highlighting the current chord badge without any cross-origin script executions.
  * Pure Streamlit column buttons allow jumping to specific beat intervals.

### 3. Build & Deployment Instability (Resolved High Risk)
* **Status**: **FULLY RESOLVED**
* **Finding**: Dependencies in [requirements.txt](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/requirements.txt) have been pinned to compatible ranges (e.g. `librosa>=0.10.0,<0.11`, `torch>=2.1`), unblocking deterministic environments.
* **Packaging**: A modern [pyproject.toml](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/pyproject.toml) configuration has been added, defining the project metadata, dependencies, dev extras (`pytest`, `ruff`, `mypy`), entry points, and linters. System imports no longer rely on `sys.path` hacks.

### 4. Fragile & Non-Deterministic Ingestion Pipeline (Resolved Medium Risk)
* **Status**: **FULLY RESOLVED**
* **Finding**: To mitigate data alignment mismatches resulting from YouTube scraping variations:
  * A duration comparison check was integrated into the ETL pipeline [etl_pipeline.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/src/etl_pipeline.py#L281-L295). If the duration delta between the loaded audio and label bounds exceeds `DURATION_TOLERANCE_SEC` (default 2.0s), a warning is logged or the song is skipped (in strict mode).
  * A standalone validation script [validate_data.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/src/validate_data.py) was added to verify shape conformity, label bounds (0-24), NaNs, and duration anomalies across all processed HDF5 files.

### 5. Lack of ML Ops & Experiment Tracking (Resolved Medium Risk)
* **Status**: **FULLY RESOLVED**
* **Finding**:
  * Hyperparameters, audio parameters, and training settings have been centralized in [config.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/src/config.py) to eliminate duplicate hardcoded values.
  * Random seed setting was centralized in [utils.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/src/utils.py#L13-L41) to enforce determinism across Python, NumPy, PyTorch, CUDA, and cuDNN. Dataloaders leverage seed parameters for song-level splits.
  * Checkpoint files are now saved with unique timestamps (e.g., `models/transformer_best_<timestamp>.pth`) to prevent accidental overwrites, and the best model is copied to the expected standard file for backward compatibility.
  * A lightweight experiment tracker [tracking.py](file:///Users/ssc1_1/Code/Chord_processing_identifier%20/src/tracking.py) records metrics to TensorBoard and standard logs.
  * Classification evaluations now calculate and output frame-level accuracy, non-N (no-chord) accuracy, and simplified chord symbol overlap (symbol recall) at every validation epoch. Completed runs generate a structured `run_config_*.json` containing all parameters and metric evaluations.

---

## 🚀 Future Roadmap & Optimizations

While the project is now stable and production-ready, the following next steps can be pursued to further optimize the system:

1. **Build Locking**:
   * Transition from pinned ranges to strict lockfiles (e.g., `uv.lock` or `poetry.lock`) to achieve absolute, reproducible dependency trees.
2. **Strict Typings**:
   * Progressively enforce strict Mypy checks (`strict = true` in `pyproject.toml`) across the codebase to capture silent type issues.
3. **Advanced MIR Evaluations**:
   * Fully integrate standard `mir_eval.chord` metrics (such as Weighted Average Overlap Ratio / WAOR and root/bass/majmin chord variants) into the training validation report once the optional `mir_eval` dependency is fully active in all environments.
4. **DSP & Spectral Resolution**:
   * Optionally increase CQT resolution from 12 to 24 or 36 bins per octave (2-3 bins per semitone) in the DSP pipeline to make the model robust against out-of-tune audio or variations.
