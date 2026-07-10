# Data Availability and Expected Input Formats

This repository provides the code needed to run Bio-PD on preprocessed or publicly available datasets. Large source datasets are not bundled with the code repository.

## 1. Single-cell transcriptomics

### Expected input

`run_biopd_cell.py` expects a preprocessed AnnData object:

```text
data/cell/<dataset>.h5ad
```

Required fields:

```python
adata.obs["stage"]          # biological stage / cell-state annotation
adata.obs["dpt_pseudotime"] # diffusion pseudotime computed during preprocessing
adata.obsm["X_diffmap"]     # diffusion-map coordinates computed during preprocessing
```

Recommended fields:

```python
adata.obsm["X_pca"]         # PCA representation; computed by the script if absent
adata.obsp["connectivities"]# neighborhood graph for graph regularization / alignment
adata.obsp["distances"]     # neighborhood distances
```

### Notes

`stage` is a biological annotation determined by the dataset or by the user's analysis design. It is used for visualization, evaluation, and proportional subsampling. In the label-free Bio-PD cell workflow, stage-dependent contrastive, ordering, compactness, and repulsion losses are disabled.

`dpt_pseudotime` and `X_diffmap` should be generated during preprocessing, for example with a standard Scanpy workflow using PCA, neighbors, diffusion maps, and DPT. Keeping preprocessing separate from `run_biopd_cell.py` preserves consistency with the original final training script, which assumes a preprocessed `.h5ad` input.


## 2. fMRI naturalistic cognition analysis

`run_biopd_fmri.py` expects Sherlock fMRI ROI time-series files and scene labels.

Suggested layout:

```text
data/fmri/
├── high_Visual_sherlock_movie.npy
├── aud_early_sherlock_movie.npy
├── early_visual_sherlock_movie.npy
├── pmc_nn_sherlock_movie.npy
└── sherlock_labels_coded_expanded.csv
```

The default ROI names in the script correspond to:

```text
HV   high_Visual_sherlock_movie.npy
EA   aud_early_sherlock_movie.npy
EV   early_visual_sherlock_movie.npy
PMC  pmc_nn_sherlock_movie.npy
```

## 3. EEG seizure-risk analysis

`run_biopd_eeg.py` expects the CHB-MIT Scalp EEG directory structure with EDF files and summary files.

Suggested layout:

```text
data_chb/
├── chb01/
│   ├── chb01-summary.txt
│   ├── chb01_01.edf
│   └── ...
├── chb02/
│   ├── chb02-summary.txt
│   ├── chb02_01.edf
│   └── ...
└── ...
```

The EEG script parses summary files to identify seizure intervals, extracts features from EDF recordings, performs cross-subject ChebGCN pretraining, generates unsupervised Bio-PD trajectories, and evaluates supervised seizure-risk warning models.
