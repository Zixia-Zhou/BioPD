# Bio-PD

Bio-PD (Biological Pattern Discovery) is a staged, self-refining manifold-learning framework for recovering trajectory structure from ordered, high-dimensional biological observations. This repository provides standalone scripts for the three analysis settings reported in the manuscript:

| Script | Analysis setting | Main input |
|---|---|---|
| `run_biopd_cell.py` | Single-cell developmental trajectories | Preprocessed `.h5ad` |
| `run_biopd_fmri.py` | Naturalistic fMRI trajectories | ROI `.npy` files and scene labels |
| `run_biopd_eeg.py` | CHB-MIT EEG seizure-risk trajectories | CHB-MIT EDF directory |

The scripts are self-contained and do not require the original project-local helper modules.

## Installation

Create a fresh environment and install the dependencies:

```bash
python -m venv biopd_env
source biopd_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`torch-geometric` can be sensitive to the installed PyTorch and CUDA versions. If the generic `pip install` route fails, install PyTorch and PyTorch Geometric following the official wheel selector for your CUDA/PyTorch combination, then install the remaining packages from `requirements.txt`.

## Quick start

### Single-cell example

```bash
bash examples/run_cell_paul15.sh
```

### fMRI example

```bash
bash examples/run_fmri_sherlock.sh
```

### EEG example

```bash
bash examples/run_eeg_chbmit.sh
```

## Output

Each script writes results to a task-specific subdirectory under `results/` by default. Typical outputs include:

- learned Bio-PD embeddings (`.npy`);
- figures (`.png` / `.pdf`, depending on the task);
- metric tables (`.csv`);
- serialized result objects where applicable.

## Reproducibility

All scripts expose random-seed arguments and set NumPy/PyTorch seeds. Minor numerical differences may occur across GPU architectures, CUDA versions, PyTorch versions, and PyTorch Geometric builds. See `REPRODUCIBILITY.md` for recommended reporting and comparison procedures.

## Data availability

The repository does not include large public datasets. Data sources, required fields, and expected file layouts are documented in `DATA_AVAILABILITY.md`.

