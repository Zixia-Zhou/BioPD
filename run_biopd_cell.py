#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bio-PD cell-level training and evaluation script.

This standalone script runs the cell-level Bio-PD workflow from a preprocessed
AnnData object. The input .h5ad file should contain biological annotations for
visualization/evaluation and precomputed diffusion-pseudotime features.

Required input fields:
  - adata.obs["stage"]: biological stage or cell-state annotation
  - adata.obs["dpt_pseudotime"]: diffusion pseudotime from preprocessing
  - adata.obsm["X_diffmap"]: diffusion-map representation from preprocessing

Example usage:
  python run_biopd_cell.py --h5ad data/cell/paul15.h5ad --dataset paul15 --seed 0 --out_dir ./results/cell
  python run_biopd_cell.py --h5ad data/cell/celegans.h5ad --dataset celegans --seed 0 --subset_k 8000 --out_dir ./results/cell
"""

import os, sys, re, glob, warnings, time, math, argparse, json
os.environ.setdefault("TMPDIR", "./tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from scipy.stats import entropy, spearmanr, kendalltau

try:
    import umap; HAS_UMAP = True
except ImportError: HAS_UMAP = False
try:
    import phate; HAS_PHATE = True
except ImportError: HAS_PHATE = False


# ##############################################################################
#  METRIC HELPERS
# ##############################################################################

def _as_categorical_series(stages):
    """Return (Series[categorical], categories(list), codes(array))."""
    if isinstance(stages, pd.Categorical):
        ser = pd.Series(stages)
        cat = ser.cat
    elif isinstance(stages, pd.Series):
        ser = stages.astype("category")
        cat = ser.cat
    else:
        ser = pd.Series(stages).astype("category")
        cat = ser.cat
    codes = cat.codes
    return ser, list(cat.categories), codes

def _knn_indices_metric(X, k=15):
    """
    Compute kNN indices for metrics (single call per embedding).
    Complexity ~ O(N * k * d).
    """
    X = np.asarray(X)
    k = min(k, max(2, X.shape[0]-1))
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(X)
    _, idx = nbrs.kneighbors(X)
    return idx[:, 1:]  # drop self

def _neighbor_stats(Y, y_idx, k=15):

    idx = _knn_indices_metric(Y, k=k)
    n = Y.shape[0]
    S = int(y_idx.max()) + 1

    ent_list = np.zeros(n, dtype=float)
    same_frac = np.zeros(n, dtype=float)
    adj_frac = np.zeros(n, dtype=float)
    far_frac = np.zeros(n, dtype=float)

    for i, nbrs in enumerate(idx):
        if nbrs.size == 0:
            continue
        labels_nbr = y_idx[nbrs]

        # entropy over neighbor stage distribution
        hist = np.bincount(labels_nbr, minlength=S).astype(float)
        p = hist / max(hist.sum(), 1.0)
        ent_list[i] = entropy(p, base=2)

        # stage differences
        diffs = np.abs(labels_nbr - y_idx[i])
        same_frac[i] = np.mean(diffs == 0)
        adj_frac[i] = np.mean(diffs <= 1)   # includes same stage
        far_frac[i] = np.mean(diffs >= 2)

    return idx, ent_list, same_frac, adj_frac, far_frac

def per_stage_metrics(Y: np.ndarray, stages_cat_like, k: int = 15):
    """
    Per-stage neighbor-based metrics:
    - nn_label_entropy_stage: mean neighbor label entropy within each stage (lower is better)
    - same_stage_frac: mean fraction of same-stage neighbors (cluster compactness, higher is better)
    - adjacent_stage_frac: mean fraction of neighbors from same or adjacent stages (trajectory continuity)
    - far_stage_frac: mean fraction of neighbors from non-adjacent stages (trajectory violations)
    Signature kept unchanged.
    """
    stages_ser, cats, codes = _as_categorical_series(stages_cat_like)
    codes_np = codes.to_numpy() if hasattr(codes, "to_numpy") else np.asarray(codes)
    if Y.shape[0] < 3:
        return pd.DataFrame(columns=[
            "stage", "n_cells",
            "nn_label_entropy_stage",
            "same_stage_frac",
            "adjacent_stage_frac",
            "far_stage_frac"
        ])

    _, ent_arr, same_frac, adj_frac, far_frac = _neighbor_stats(Y, codes_np, k=k)
    S = int(codes_np.max()) + 1

    sizes = np.bincount(codes_np, minlength=S)
    per_rows = []
    for s in range(S):
        m = (codes_np == s)
        n_s = int(m.sum())
        if n_s == 0:
            per_rows.append([s, 0, np.nan, np.nan, np.nan, np.nan])
        else:
            per_rows.append([
                s,
                n_s,
                float(np.nanmean(ent_arr[m])),
                float(np.nanmean(same_frac[m])),
                float(np.nanmean(adj_frac[m])),
                float(np.nanmean(far_frac[m])),
            ])

    per_df = pd.DataFrame(
        per_rows,
        columns=[
            "stage_idx",
            "n_cells",
            "nn_label_entropy_stage",
            "same_stage_frac",
            "adjacent_stage_frac",
            "far_stage_frac",
        ],
    )

    # map index to stage name
    stage_names = pd.Categorical(stages_ser).categories
    per_df["stage"] = per_df["stage_idx"].map(
        lambda i: str(stage_names[i]) if 0 <= i < len(stage_names) else f"{i}"
    )
    return per_df.drop(columns=["stage_idx"])

def global_metrics(Y: np.ndarray, dpt: np.ndarray, stages_cat_like, random_state: int = 0):

    stages_ser, cats, codes = _as_categorical_series(stages_cat_like)
    y_idx = codes.to_numpy() if hasattr(codes, "to_numpy") else np.asarray(codes)
    if Y.shape[0] < 4 or len(np.unique(y_idx)) < 2:
        return {
            "nn_label_entropy": np.nan,
            "same_stage_fraction": np.nan,
            "nn_stage_adjacency_leq1": np.nan,
            "boundary_sharpness_gap2": np.nan,
            "traj_spearman": np.nan,
            "traj_kendall": np.nan,
        }

    # neighbor-based cluster / trajectory stats
    _, ent_arr, same_frac, adj_frac, far_frac = _neighbor_stats(
        Y, y_idx, k=min(15, max(2, Y.shape[0]-1))
    )
    nn_ent = float(np.nanmean(ent_arr))
    same_stage_fraction = float(np.nanmean(same_frac))
    nn_adj = float(np.nanmean(adj_frac))
    far_mean = float(np.nanmean(far_frac))
    sharp = 1.0 - far_mean  # higher => fewer non-adjacent neighbor links

    # trajectory monotonicity wrt dpt via 1D PCA of embedding
    dpt = np.asarray(dpt, dtype=float)
    if np.all(np.isnan(dpt)) or np.nanstd(dpt) < 1e-8:
        spc, knd = np.nan, np.nan
    else:
        try:
            pc1 = PCA(n_components=1, random_state=random_state).fit_transform(Y)[:, 0]
            spc = float(spearmanr(dpt, pc1, nan_policy="omit").statistic)
            knd = float(kendalltau(dpt, pc1).statistic)
        except Exception:
            spc, knd = np.nan, np.nan

    return {
        "nn_label_entropy": nn_ent,
        "same_stage_fraction": same_stage_fraction,
        "nn_stage_adjacency_leq1": nn_adj,
        "boundary_sharpness_gap2": sharp,
        "traj_spearman": spc,
        "traj_kendall": knd,
    }

def _fixed_stage_palette(categories):
    base = sc.pl.palettes.default_20 + sc.pl.palettes.default_102
    return base[:len(categories)]


# ##############################################################################
#  CONFIGURATION
# ##############################################################################

class Cfg:
    cuda_device = "1"
    seed = 0
    max_epochs = 200
    patience = 30
    learning_rate = 1e-3
    weight_decay = 1e-5
    max_grad_norm = 5.0
    batch_size = 2048
    context = 256
    stride = None
    center_nonoverlap = True
    low_dim = 2
    perplexity = 50
    HD_type = "sherlock"
    lambda_graph = 0.25
    graph_center_only = True
    dense_units = (1024, 512, 256)
    final_units = 2
    dropout = 0.0
    n_recur = 4
    refresh_P_every_epochs_late = 10
    warmup_epochs = 10
    refresh_P_every_epochs_warm = 2
    use_amp = True
    try_compile = True
    msd_dims = 20
    msd_scales = (1, 2, 4, 8)
    knn_k = 30
    use_distance_weight = False
    mix_perplexities = (20, 50, 120)
    density_equalize_weight = 0.5
    P_backend = "cached_knn"
    P_knn_k = 150
    P_beta_tol = 1e-5
    P_beta_max_iter = 50
    debug_mode = False
    check_gradients = False
    benchmark_on_start = True
    lambda_contrastive = 0.1
    contrastive_n_samples = 512
    contrastive_margin = 1.0
    # adaptive fields
    kl_alpha = None              # None = default (low_dim-1); 0.5 for small data
    early_exag_factor = 1.0      # multiply P in early epochs (1.0 = no exag)
    early_exag_epochs = 0        # number of epochs for early exaggeration
    lambda_ordering = 0.0        # centroid ordering loss weight
    contrastive_stages = ("Dense3",)  # which stages get contrastive loss
    data_dir = "./data/cell"
    out_dir = "./results/cell"


def adapt_cfg_to_dataset_size(cfg, N):
    """Adapt hyperparameters based on dataset size."""
    import math

    if N < 5000:
        # ---- SMALL DATA MODE ----
        cfg.batch_size = N
        cfg.context = 0
        cfg.center_nonoverlap = True

        cfg.dense_units = (1024, 512, 256)
        cfg.dropout = 0.10

        cfg.mix_perplexities = (10, 30, 50)
        cfg.perplexity = 30

        cfg.P_knn_k = min(100, max(15, N // 20))
        cfg.knn_k = max(5, min(20, int(math.sqrt(N) * 0.3)))

        cfg.lambda_contrastive = 0.3
        cfg.contrastive_margin = 1.0
        cfg.contrastive_n_samples = 1024
        cfg.contrastive_stages = ("Dense1", "Dense2", "Dense3")
        cfg.lambda_ordering = 0.5

        cfg.lambda_graph = 0.5
        cfg.graph_center_only = False
        cfg.density_equalize_weight = 0.7

        cfg.kl_alpha = 0.5
        cfg.early_exag_factor = 8.0
        cfg.early_exag_epochs = 50

        cfg.patience = 60
        cfg.max_epochs = 500
        cfg.learning_rate = 5e-4
        cfg.refresh_P_every_epochs_late = 2
        cfg.refresh_P_every_epochs_warm = 1
        cfg.warmup_epochs = 30

    elif N < 20000:
        # ---- MEDIUM DATA MODE ----
        cfg.batch_size = N
        cfg.context = 0
        cfg.center_nonoverlap = True

        cfg.dense_units = (1024, 512, 256)
        cfg.dropout = 0.05

        cfg.mix_perplexities = (15, 40, 80)
        cfg.perplexity = 40

        cfg.P_knn_k = min(150, max(30, N // 50))
        cfg.knn_k = max(5, min(30, int(math.sqrt(N) * 0.5)))

        cfg.lambda_contrastive = 0.2
        cfg.contrastive_n_samples = 1024
        cfg.contrastive_stages = ("Dense2", "Dense3")
        cfg.lambda_ordering = 0.3

        cfg.lambda_graph = 0.35
        cfg.graph_center_only = False
        cfg.density_equalize_weight = 0.6

        cfg.kl_alpha = 0.75
        cfg.early_exag_factor = 4.0
        cfg.early_exag_epochs = 30

        cfg.patience = 50
        cfg.max_epochs = 400
        cfg.learning_rate = 7e-4
        cfg.refresh_P_every_epochs_late = 5
        cfg.refresh_P_every_epochs_warm = 1
        cfg.warmup_epochs = 20

    # else: N >= 20000 — keep default large-data settings
    # contrastive_stages stays ("Dense3",), lambda_ordering=0, early_exag=1.0

    print(f"[ADAPTIVE CONFIG] N={N:,}")
    print(f"  dense_units={cfg.dense_units}, dropout={cfg.dropout:.2f}")
    print(f"  batch_size={cfg.batch_size}, context={cfg.context}")
    print(f"  mix_perplexities={cfg.mix_perplexities}, P_knn_k={cfg.P_knn_k}, knn_k={cfg.knn_k}")
    print(f"  lambda_contrastive={cfg.lambda_contrastive}, lambda_graph={cfg.lambda_graph}, "
          f"lambda_ordering={cfg.lambda_ordering}")
    print(f"  contrastive_stages={cfg.contrastive_stages}")
    print(f"  kl_alpha={cfg.kl_alpha}, early_exag={cfg.early_exag_factor}x for {cfg.early_exag_epochs} epochs")
    print(f"  patience={cfg.patience}, max_epochs={cfg.max_epochs}, lr={cfg.learning_rate}")
    return cfg


# ##############################################################################
#  HIGH-DIMENSIONAL PROBABILITY HELPERS
# ##############################################################################

# ------------------------------------------------------------------------------
# Embedded high-dimensional probability helpers
# ------------------------------------------------------------------------------
# The original probability helper functions are embedded here so this script is
# self-contained while preserving the same implementation path.
import numpy as np
import sklearn
import numpy as Math
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

try:
    import numba
    from numba import njit
except Exception:
    numba = None
    def njit(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func


@njit
def Hbeta(D, beta):
    P = np.exp(-D * beta)  
    sumP = np.float32(0.0)
    for i in range(P.shape[0]):
        sumP += P[i]
    sumP = sumP + np.float32(1e-8)

    DP = np.float32(0.0)
    for i in range(P.shape[0]):
        DP += np.float32(D[i] * P[i])

    H = np.log(sumP) + beta * DP / sumP

    for i in range(P.shape[0]):
        P[i] = P[i] / sumP

    return H, P


# This function prepares the distance matrix for the numba-optimized function
def prepare_distances(X):
    """Calculate squared Euclidean distances"""
    return squareform(pdist(X, 'sqeuclidean'))


def x2p2(X, tol=1e-5, perplexity=30.0):
    n, d = X.shape
    D = np.square(sklearn.metrics.pairwise_distances(X, metric='euclidean'))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        Di = D[i]
        (H, thisP) = Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            #  while  tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            (H, thisP) = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        P[i, idx[i]] = thisP

    return P


def x2p1(X=np.array([]), tol=1e-5, perplexity=30.0):
    (n, d) = X.shape
    D = np.square(sklearn.metrics.pairwise_distances(X, metric='euclidean'))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        Di = D[i]
        (H, thisP) = Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            #  while tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            (H, thisP) = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        P[i, idx[i]] = thisP

    return P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    (n, d) = X.shape
    D = np.square(sklearn.metrics.pairwise_distances(X, metric='euclidean'))
    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        Di = D[i]
        (H, thisP) = Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0
        # while np.abs(Hdiff) > tol and tries < 50:
        while tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            (H, thisP) = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        P[i, idx[i]] = thisP

    return P


@njit
def x2p_optimized(D, perplexity=30.0, tol=1e-5, max_tries=50):
    """
    Optimized version of x2p using numba
    D should be a pre-computed squared distance matrix
    """
    n = D.shape[0]
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Get distances from point i to all other points
        conditional_indices = np.zeros(n - 1, dtype=np.int64)
        counter = 0
        for j in range(n):
            if i != j:
                conditional_indices[counter] = j
                counter += 1

        Di = np.zeros(n - 1)
        for idx in range(n - 1):
            j = conditional_indices[idx]
            Di[idx] = D[i, j]

        # Initialize some variables
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0

        # Binary search for the correct value of beta
        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries = 0

        # Binary search loop
        while np.abs(Hdiff) > tol and tries < max_tries:
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            # Compute new values
            H, thisP = Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        for idx in range(len(conditional_indices)):
            j = conditional_indices[idx]
            P[i, j] = thisP[idx]

    return P

# ##############################################################################
#  STAGE AND INPUT HELPERS
# ##############################################################################

CANONICAL_STAGES = [
    "Initial gastrulation", "Mid gastrulation",
    "Early neurula", "Late neurula",
    "Initial tailbud I", "Early tailbud I", "Mid tailbud I", "Late tailbud I",
    "Early tailbud II", "Mid tailbud II", "Late tailbud II",
    "Larva",
]
_STAGE_TOKENS = [
    ("lattii","Late tailbud II"),("midtii","Mid tailbud II"),("eartii","Early tailbud II"),("tii","Late tailbud II"),
    ("latti","Late tailbud I"),("midti","Mid tailbud I"),("earti","Early tailbud I"),("initi","Initial tailbud I"),("ti1","Late tailbud I"),
    ("latn","Late neurula"),("earn","Early neurula"),
    ("midg","Mid gastrulation"),("inig","Initial gastrulation"),("larva","Larva")
]
_REP_PAT = re.compile(r"(rep\d+)", re.IGNORECASE)

def _canon_stage_from_name(name: str) -> str:
    base = os.path.basename(name).lower()
    s = base.replace("-", "").replace("_", "")
    for key, stage in sorted(_STAGE_TOKENS, key=lambda kv: -len(kv[0])):
        if key in s:
            return stage
    return "unknown"

def infer_stage_and_rep(path_or_file: str):
    stage = _canon_stage_from_name(path_or_file)
    base = os.path.basename(path_or_file).lower()
    m = _REP_PAT.search(base)
    rep = m.group(1).lower() if m else "rep?"
    return stage, rep


# ##############################################################################
#  INPUT/OUTPUT HELPERS
# ##############################################################################

def discover_inputs(root: str):
    files = sorted(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))
    files += sorted(glob.glob(os.path.join(root, "**", "*.h5ad"), recursive=True))
    inputs = files[:]
    if not inputs:
        mtx_files = sorted(
            glob.glob(os.path.join(root, "**", "matrix.mtx"), recursive=True)
            + glob.glob(os.path.join(root, "**", "matrix.mtx.gz"), recursive=True)
        )
        inputs = sorted({os.path.dirname(p) for p in mtx_files})
    return inputs

def read_counts_from_path(path: str) -> ad.AnnData:
    if os.path.isfile(path):
        if path.lower().endswith(".h5"):
            try:
                adata = sc.read_10x_h5(path)
            except Exception:
                adata = sc.read_h5ad(path)
        elif path.lower().endswith(".h5ad"):
            adata = sc.read_h5ad(path)
        else:
            raise RuntimeError(f"Unsupported file type: {path}")
    else:
        mtx_dir = None
        for cand in ("matrix.mtx","matrix.mtx.gz"):
            cand_p = os.path.join(path, cand)
            if os.path.exists(cand_p):
                mtx_dir = path
                break
        if mtx_dir is None:
            raise RuntimeError(f"No 10x mtx found in dir: {path}")
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    stage, rep = infer_stage_and_rep(path)
    adata.obs["stage"] = stage
    adata.obs["rep"]   = rep
    adata.obs["batch"] = f"{stage}_{rep}"
    return adata

def print_stage_counts(adata: ad.AnnData, header: str):
    print(f"[STAGE COUNTS] {header}")
    if "stage" not in adata.obs:
        for s in CANONICAL_STAGES:
            print(f"  {s:20s}: 0")
        return
    vc = adata.obs["stage"].astype(str).value_counts(dropna=False)
    for s in CANONICAL_STAGES:
        print(f"  {s:20s}: {int(vc.get(s, 0))}")


# ##############################################################################
#  DATA SANITIZATION
# ##############################################################################

def _nanfix_matrix(X):
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.nan_to_num(X.data, copy=False)
        return X
    X = np.asarray(X)
    if not X.flags.writeable:
        X = X.copy()
    return np.nan_to_num(X, copy=False)

def sanitize_adata(adata: ad.AnnData):
    adata.X = _nanfix_matrix(adata.X)
    if hasattr(adata, "layers") and adata.layers:
        for k in list(adata.layers.keys()):
            adata.layers[k] = _nanfix_matrix(adata.layers[k])
    if hasattr(adata, "obsm"):
        for k in list(adata.obsm.keys()):
            arr = adata.obsm[k]
            if isinstance(arr, np.ndarray):
                np.nan_to_num(arr, copy=False)
            elif sp.issparse(arr):
                adata.obsm[k] = _nanfix_matrix(arr)
    return adata


# ##############################################################################
#  HIGH-DIMENSIONAL AFFINITY COMPUTATION
# ##############################################################################

class FastPComputer:
    def __init__(self, cfg: Cfg, device):
        self.cfg = cfg
        self.device = device
        self.knn_cache = {}
    def compute_P_mixture(self, X: np.ndarray, perplexities: tuple,
                          deg_local: Optional[np.ndarray] = None,
                          cache_key: Optional[str] = None) -> np.ndarray:
        n = X.shape[0]
        if n <= 2:
            P = np.eye(n, dtype=np.float32)
            return P / (P.sum() + 1e-8)
        if self.cfg.P_backend == "cached_knn":
            return self._compute_P_cached_knn(X, perplexities, deg_local, cache_key)
        elif self.cfg.P_backend == "torch_exact":
            return self._compute_P_torch_exact(X, perplexities, deg_local, cache_key)
        elif self.cfg.P_backend == "original":
            return self._compute_P_original(X, perplexities, deg_local)
        else:
            return self._compute_P_cached_knn(X, perplexities, deg_local, cache_key)
    def _compute_P_original(self, X, perplexities, deg_local):
        Ps = [x2p(X, perp) for perp in perplexities]
        P = np.mean(Ps, axis=0)
        P[np.isnan(P)] = 0.0
        P = 0.5*(P+P.T)
        return self._apply_density_equalization(P, deg_local)
    def _compute_P_cached_knn(self, X, perplexities, deg_local, cache_key):
        n = X.shape[0]
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if cache_key and cache_key in self.knn_cache:
            idx, d2 = self.knn_cache[cache_key]
        else:
            k = min(self.cfg.P_knn_k, max(1, n-1))
            if n < 5000:
                idx, d2 = self._gpu_knn(X_t, k)
            else:
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto")
                nbrs.fit(X)
                dists, idx_np = nbrs.kneighbors(X)
                idx = torch.as_tensor(idx_np[:, 1:], dtype=torch.long, device=self.device)
                d2  = torch.as_tensor(dists[:, 1:]**2, dtype=torch.float32, device=self.device)
            if cache_key:
                self.knn_cache[cache_key] = (idx, d2)
        P_acc = torch.zeros_like(d2)
        for perp in perplexities:
            k_ref = min(d2.shape[1]-1, max(1, int(0.75*perp)))
            d2_sorted, _ = torch.sort(d2, dim=1)
            sigma2 = d2_sorted[:, k_ref].unsqueeze(1).clamp(min=1e-8)
            P_k = torch.exp(-d2/(2.0*sigma2))
            P_k = P_k/(P_k.sum(dim=1, keepdim=True)+1e-12)
            P_acc += P_k
        P_acc = P_acc/len(perplexities)
        P_dense = self._sparse_to_dense_P(idx, P_acc, n)
        return self._apply_density_equalization(P_dense, deg_local)
    def _compute_P_torch_exact(self, X, perplexities, deg_local, cache_key):
        n = X.shape[0]
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        k = min(self.cfg.P_knn_k, max(1, n-1))
        if n < 5000:
            idx, d2 = self._gpu_knn(X_t, k)
        else:
            nbrs = NearestNeighbors(n_neighbors=k+1)
            nbrs.fit(X)
            dists, idx_np = nbrs.kneighbors(X)
            idx = torch.as_tensor(idx_np[:, 1:], dtype=torch.long, device=self.device)
            Xi2 = (X_t*X_t).sum(dim=1, keepdim=True)
            Xnbr = X_t[idx]
            Xnbr2 = (Xnbr*Xnbr).sum(dim=2)
            dot = (X_t[:, None, :]*Xnbr).sum(dim=2)
            d2 = torch.clamp(Xi2 + Xnbr2 - 2.0*dot, min=0)
        P_acc = torch.zeros_like(d2)
        for perp in perplexities:
            logU = float(np.log(perp))
            beta = torch.ones((n,1), device=self.device)
            beta_min = torch.full((n,1), -float("inf"), device=self.device)
            beta_max = torch.full((n,1),  float("inf"), device=self.device)
            for _ in range(self.cfg.P_beta_max_iter):
                P = torch.exp(-d2*beta)
                sumP = P.sum(dim=1, keepdim=True) + 1e-12
                H = torch.log(sumP) + beta*(d2*P).sum(dim=1, keepdim=True)/sumP
                Hdiff = H - logU
                done = Hdiff.abs() < self.cfg.P_beta_tol
                if done.all(): break
                pos = Hdiff > 0
                beta_min = torch.where(pos, beta, beta_min)
                beta_max = torch.where(~pos, beta, beta_max)
                beta_new = torch.where(
                    torch.isinf(beta_min), beta/2.0,
                    torch.where(torch.isinf(beta_max), beta*2.0,
                                torch.where(pos, 0.5*(beta+beta_max), 0.5*(beta+beta_min)))
                )
                beta = torch.where(done, beta, beta_new)
            P = torch.exp(-d2*beta)
            P = P/(P.sum(dim=1, keepdim=True)+1e-12)
            P_acc += P
        P_acc = P_acc/len(perplexities)
        P_dense = self._sparse_to_dense_P(idx, P_acc, n)
        return self._apply_density_equalization(P_dense, deg_local)
    def _gpu_knn(self, X: torch.Tensor, k: int):
        n = X.shape[0]
        X_norm = (X*X).sum(dim=1, keepdim=True)
        d2 = torch.clamp(X_norm + X_norm.t() - 2.0*(X@X.t()), min=0)
        d2.fill_diagonal_(float("inf"))
        d2, idx = torch.topk(d2, k, dim=1, largest=False)
        return idx, d2
    def _sparse_to_dense_P(self, idx, vals, n):
        k = idx.shape[1]
        rows = torch.arange(n, device=self.device).unsqueeze(1).repeat(1, k).reshape(-1)
        cols = idx.reshape(-1)
        vals_flat = vals.reshape(-1)
        P_sparse = sp.coo_matrix((vals_flat.cpu().numpy(),
                                  (rows.cpu().numpy(), cols.cpu().numpy())), shape=(n, n))
        P_sparse = 0.5*(P_sparse + P_sparse.T)
        P_dense = P_sparse.todense().A.astype(np.float32)
        P_dense = P_dense/(P_dense.sum()+1e-8)
        return np.maximum(P_dense, 1e-12)
    def _apply_density_equalization(self, P, deg_local):
        if deg_local is None or Cfg.density_equalize_weight < 1e-8:
            P = P/(P.sum()+1e-8)
            return np.maximum(P, 1e-12).astype(np.float32)
        d = deg_local.astype(np.float32) + 1e-6
        w = 1.0/d
        W = np.outer(w, w)
        W = W/(W.max()+1e-8)
        alpha = Cfg.density_equalize_weight
        P = P*((1.0-alpha) + alpha*W)
        P = P/(P.sum()+1e-8)
        return np.maximum(P, 1e-12).astype(np.float32)
    def clear_cache(self): self.knn_cache.clear()


# ##############################################################################
#  MODEL AND LOSSES
# ##############################################################################

class MLPBioPD(nn.Module):
    def __init__(self, in_dim: int, hidden=(1024, 512, 256), out_dim=2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.out = nn.Linear(hidden[2], out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z1 = self.drop(self.act(self.fc1(x)))
        z2 = self.drop(self.act(self.fc2(z1)))
        z3 = self.drop(self.act(self.fc3(z2)))
        return self.out(z3)
    @torch.no_grad()
    def get_layer_output(self, x, L: int):
        z1 = self.act(self.fc1(x))
        if L == 1: return z1
        z2 = self.act(self.fc2(z1))
        if L == 2: return z2
        z3 = self.act(self.fc3(z2))
        if L == 3: return z3
        raise ValueError("L must be 1/2/3")

def create_kl_divergence_stable_weighted(low_dim=2, alpha_override=None):
    def KLdiv(P, Y, row_weights=None):
        alpha = alpha_override if alpha_override is not None else (low_dim - 1)
        eps = 1e-15
        sumY = torch.sum(Y**2, dim=1)
        D = torch.clamp(sumY[:, None] + sumY[None, :] - 2 * (Y @ Y.t()), min=0)
        Q = torch.pow(1 + D/alpha, -(alpha+1)/2)
        mask = 1 - torch.eye(Y.shape[0], device=Y.device, dtype=Q.dtype)
        Q = Q * mask
        Q = Q / (torch.sum(Q) + eps)
        Q = torch.clamp(Q, min=eps, max=1.0)
        sel = (P > eps)
        term = torch.where(sel, P * torch.log(P/(Q+eps)), torch.zeros_like(P))
        if row_weights is not None:
            rw = row_weights.view(-1, 1)
            return torch.sum(rw * term)
        else:
            return torch.sum(term)
    return KLdiv

def graph_smoothness_loss(Y, rows, cols, vals):
    if rows.numel() == 0:
        return Y.new_tensor(0.0)
    diffs = Y[rows] - Y[cols]
    e2 = (diffs * diffs).sum(dim=1)
    return (e2 * vals).mean()


def contrastive_trajectory_loss(Y, stage_codes, margin=1.0, n_samples=512):
    """
    Soft contrastive loss for trajectory ordering.
    Same-stage pairs: minimize distance.
    Different-stage pairs: distance should be >= margin * |stage_diff|.
    Samples random pairs for efficiency.
    """
    n = Y.shape[0]
    if n < 4:
        return Y.new_tensor(0.0)
    idx_a = torch.randint(0, n, (n_samples,), device=Y.device)
    idx_b = torch.randint(0, n, (n_samples,), device=Y.device)
    mask_diff = idx_a != idx_b
    idx_a, idx_b = idx_a[mask_diff], idx_b[mask_diff]
    if idx_a.shape[0] == 0:
        return Y.new_tensor(0.0)

    dists = torch.sqrt(torch.sum((Y[idx_a] - Y[idx_b]) ** 2, dim=1) + 1e-8)
    stage_diffs = torch.abs(stage_codes[idx_a].float() - stage_codes[idx_b].float())

    same_mask = stage_diffs == 0
    loss_same = dists[same_mask].mean() if same_mask.any() else Y.new_tensor(0.0)

    diff_mask = stage_diffs > 0
    if diff_mask.any():
        target_dist = margin * stage_diffs[diff_mask]
        loss_diff = torch.clamp(target_dist - dists[diff_mask], min=0).mean()
    else:
        loss_diff = Y.new_tensor(0.0)

    return loss_same + loss_diff


def centroid_ordering_loss(Y, stage_codes, n_stages):
    """Encourage stage centroids to be monotonically ordered along the main trajectory axis.
    Computes centroids per stage, projects onto the first-to-last direction,
    penalizes violations of monotonic ordering.
    """
    centroids = []
    valid_stages = []
    for s in range(n_stages):
        mask = stage_codes == s
        if mask.sum() >= 2:
            centroids.append(Y[mask].mean(0))
            valid_stages.append(s)
    if len(centroids) < 3:
        return Y.new_tensor(0.0)
    centroids = torch.stack(centroids)
    direction = centroids[-1] - centroids[0]
    direction = direction / (torch.norm(direction) + 1e-8)
    projections = centroids @ direction
    diffs = projections[1:] - projections[:-1]
    violations = torch.clamp(-diffs, min=0)
    return violations.mean()


def pseudotime_flow_loss(Y, dpt_t):
    """Differentiable correlation between embedding PC1 and DPT.
    Targets Spearman_rho metric. From agent2."""
    if Y.shape[0] < 10:
        return Y.new_tensor(0.0)
    Y_c = Y - Y.mean(0)
    try:
        _, _, Vt = torch.linalg.svd(Y_c, full_matrices=False)
        proj = Y_c @ Vt[0]
    except Exception:
        proj = Y_c[:, 0]
    proj_z = (proj - proj.mean()) / (proj.std() + 1e-8)
    dpt_z = (dpt_t - dpt_t.mean()) / (dpt_t.std() + 1e-8)
    corr = (proj_z * dpt_z).mean()
    return 1.0 - corr.abs()


def dpt_knn_smoothness_loss(Y, dpt_t, k=15, n_sample=4096):
    """Encourage kNN neighbors in embedding to have similar DPT.
    Targets DDC and Trajectory_Continuity. From agent2."""
    N = Y.shape[0]
    if N < k + 1:
        return Y.new_tensor(0.0)
    if N > n_sample:
        idx = torch.randperm(N, device=Y.device)[:n_sample]
        Y_sub, dpt_sub = Y[idx], dpt_t[idx]
    else:
        Y_sub, dpt_sub = Y, dpt_t
    D = torch.cdist(Y_sub, Y_sub)
    _, knn_idx = D.topk(k + 1, largest=False)
    knn_idx = knn_idx[:, 1:]
    dpt_center = dpt_sub.unsqueeze(1).expand(-1, k)
    dpt_neighbors = dpt_sub[knn_idx]
    dpt_diff_sq = (dpt_center - dpt_neighbors) ** 2
    knn_dists = D.gather(1, knn_idx)
    weights = 1.0 / (knn_dists + 1e-8)
    weights = weights / weights.sum(1, keepdim=True)
    return (weights * dpt_diff_sq).sum(1).mean()


def cluster_compactness_loss(Y, codes, n_stages):
    """Penalize within-cluster variance (preserve Silhouette). From agent2."""
    if n_stages < 2:
        return Y.new_tensor(0.0)
    global_var = Y.var(0).sum() + 1e-8
    intra_var = Y.new_tensor(0.0)
    count = 0
    for s in range(n_stages):
        mask = (codes == s)
        if mask.sum() < 2:
            continue
        intra_var = intra_var + Y[mask].var(0).sum()
        count += 1
    if count == 0:
        return Y.new_tensor(0.0)
    return (intra_var / count) / global_var


def dimensional_spread_loss(Y, max_aspect=5.0):
    """Penalize PCA aspect ratio exceeding max_aspect. Prevents 1D collapse.

    Uses eigenvalues of the covariance matrix (PCA aspect ratio), NOT
    coordinate variances, because the output layer can learn rotations
    that make coordinate variances balanced while actual data is elongated.
    """
    Y_c = Y - Y.mean(dim=0)
    cov = Y_c.T @ Y_c / (Y.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov)  # sorted ascending
    aspect = eigvals[-1] / (eigvals[0] + 1e-8)
    return torch.clamp(aspect / max_aspect - 1.0, min=0.0)


def global_alignment_finetune(model, X_full, stage_codes, dpt_norm, adj_sparse, cfg, device,
                              n_epochs=200, lr=5e-4, lambda_ord=1.0, lambda_contr=0.3,
                              lambda_graph=0.2, lambda_flow=0.3, lambda_dpt_smooth=0.2,
                              lambda_compact=0.3, lambda_spread=0.0, spread_max_aspect=5.0,
                              batch_size=4096, n_edge_samples=30000):

    import scipy.sparse as sp

    model.eval()
    model.to(device)

    # Save original output layer weights for rollback
    out_weight_orig = model.out.weight.data.clone()
    out_bias_orig = model.out.bias.data.clone()

    # Freeze all layers except output
    for name, param in model.named_parameters():
        param.requires_grad = ('out.' in name)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[GLOBAL ALIGN] Finetune output layer only ({n_trainable} params)")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    N = X_full.shape[0]
    codes_t = torch.as_tensor(stage_codes, dtype=torch.long, device=device)
    dpt_t = torch.as_tensor(dpt_norm, dtype=torch.float32, device=device)
    n_stages = int(codes_t.max().item()) + 1

    print(f"  lambda_ord={lambda_ord}, lambda_contr={lambda_contr}, lambda_graph={lambda_graph}")
    print(f"  lambda_flow={lambda_flow}, lambda_dpt_smooth={lambda_dpt_smooth}, lambda_compact={lambda_compact}")

    # Pre-compute hidden features (fixed, no grad) — chunked to avoid OOM
    torch.cuda.empty_cache()
    chunk_size = 50000
    with torch.no_grad():
        H_chunks = []
        for i in range(0, N, chunk_size):
            chunk = torch.as_tensor(X_full[i:i+chunk_size], dtype=torch.float32, device=device)
            H_chunks.append(model.get_layer_output(chunk, 3).clone())
            del chunk
        H = torch.cat(H_chunks, dim=0)
        del H_chunks
        torch.cuda.empty_cache()

    # Dense3 embedding (before alignment) for best-of comparison
    with torch.no_grad():
        Y_dense3 = model.out(H).cpu().numpy()

    # Pre-extract graph edges if available
    if adj_sparse is not None and sp.issparse(adj_sparse):
        rows_all, cols_all = adj_sparse.nonzero()
        vals_all = np.array(adj_sparse[rows_all, cols_all]).flatten().astype(np.float32)
    else:
        rows_all = cols_all = vals_all = None

    best_loss = float('inf')
    patience_counter = 0
    patience = 40

    for ep in range(n_epochs):
        model.train()

        Y_full = model.out(H)

        # 1. Centroid ordering loss on FULL dataset
        L_ord = centroid_ordering_loss(Y_full, codes_t, n_stages)

        # 2. Contrastive loss on sampled batch
        batch_idx = torch.randint(0, N, (min(batch_size, N),), device=device)
        Y_batch = Y_full[batch_idx]
        codes_batch = codes_t[batch_idx]
        L_contr = contrastive_trajectory_loss(Y_batch, codes_batch, margin=1.0, n_samples=1024)

        # 3. Graph smoothness on sampled edges (v91: 3x more edges for PGS)
        L_graph = Y_full.new_tensor(0.0)
        if rows_all is not None:
            n_edges = len(rows_all)
            n_sample = min(n_edges, n_edge_samples)
            if n_edges > n_sample:
                edge_idx = np.random.choice(n_edges, n_sample, replace=False)
            else:
                edge_idx = np.arange(n_edges)
            r = torch.as_tensor(rows_all[edge_idx], dtype=torch.long, device=device)
            c = torch.as_tensor(cols_all[edge_idx], dtype=torch.long, device=device)
            v = torch.as_tensor(vals_all[edge_idx], dtype=torch.float32, device=device)
            L_graph = graph_smoothness_loss(Y_full, r, c, v)

        # 4. Pseudotime flow loss (PC1-DPT correlation)
        L_flow = pseudotime_flow_loss(Y_full, dpt_t) if lambda_flow > 0 else Y_full.new_tensor(0.0)

        # 5. DPT kNN smoothness (every 5 epochs — expensive due to cdist)
        L_dpt_sm = Y_full.new_tensor(0.0)
        if lambda_dpt_smooth > 0 and (ep % 5 == 0):
            L_dpt_sm = dpt_knn_smoothness_loss(Y_full, dpt_t, k=15, n_sample=min(4096, N))

        # 6. Cluster compactness
        L_compact = cluster_compactness_loss(Y_full, codes_t, n_stages) if lambda_compact > 0 else Y_full.new_tensor(0.0)

        # 7. Dimensional spread (VICReg-style: prevents 1D collapse)
        L_spread = dimensional_spread_loss(Y_full, max_aspect=spread_max_aspect) if lambda_spread > 0 else Y_full.new_tensor(0.0)

        loss = (lambda_ord * L_ord + lambda_contr * L_contr + lambda_graph * L_graph +
                lambda_flow * L_flow + lambda_dpt_smooth * L_dpt_sm + lambda_compact * L_compact +
                lambda_spread * L_spread)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        opt.step()
        sch.step()

        loss_val = float(loss.item())
        if (ep + 1) % 20 == 0 or ep == 0:
            spread_str = f", spread={float(L_spread):.4f}" if lambda_spread > 0 else ""
            print(f"  [GLOBAL ALIGN] Epoch {ep+1}/{n_epochs} | loss={loss_val:.4f} "
                  f"(ord={float(L_ord):.4f}, contr={float(L_contr):.4f}, graph={float(L_graph):.4f}, "
                  f"flow={float(L_flow):.4f}, dpt_sm={float(L_dpt_sm):.4f}, compact={float(L_compact):.4f}{spread_str})")

        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [GLOBAL ALIGN] Early stop at epoch {ep+1}")
                break

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    with torch.no_grad():
        Y_aligned = model.out(H).cpu().numpy()

    # ---- BEST-OF SELECTION: compare Dense3 vs Aligned via metric_stage_order ----
    codes_np = stage_codes.copy()
    n_stg = int(codes_np.max()) + 1
    score_d3 = metric_stage_order(Y_dense3, codes_np, n_stg)
    score_al = metric_stage_order(Y_aligned, codes_np, n_stg)

    print(f"[BEST-OF] Dense3 Stage_Order={score_d3:.4f}, Aligned Stage_Order={score_al:.4f}")

    if score_al >= score_d3 - 0.02:  # Use aligned unless it clearly hurts
        print(f"[BEST-OF] -> Using ALIGNED embedding")
        Y_best = Y_aligned
        selected = "aligned"
    else:
        print(f"[BEST-OF] -> Reverting to DENSE3 (alignment hurt ordering by {score_d3 - score_al:.4f})")
        # Restore original output layer weights
        model.out.weight.data.copy_(out_weight_orig)
        model.out.bias.data.copy_(out_bias_orig)
        Y_best = Y_dense3
        selected = "dense3_reverted"

    print(f"[GLOBAL ALIGN] Done. Selected={selected}, shape={Y_best.shape}")
    return Y_best, selected, Y_dense3, Y_aligned


# ##############################################################################
#  PIPELINE UTILITIES
# ##############################################################################

def setup_env(cfg: Cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_windows(N: int, B: int, C: int, S: int) -> List[Tuple[int, int, int, int]]:
    wins, i = [], 0
    while i < N:
        st = i
        ed = min(i + B + 2*C, N)
        cst = min(st + C, ed)
        ced = min(cst + B, ed)
        if ed - st >= 2 and ced - cst >= 2:
            wins.append((st, ed, cst, ced))
        i += S
    return wins

def _knn_on_diffusion(psi_win: np.ndarray, k: int, use_dist_w: bool):
    k_eff = min(k, max(2, psi_win.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nbrs.fit(psi_win)
    dists, idx = nbrs.kneighbors(psi_win, return_distance=True)
    Nw = psi_win.shape[0]
    rows, cols, vals = [], [], []
    if use_dist_w:
        med = np.median(dists[:, 1:]) + 1e-8
        s2 = med * med
        w_local = np.exp(-((dists ** 2) / (s2 + 1e-12)))
    else:
        w_local = np.ones_like(dists)
    for i in range(Nw):
        for jpos in range(1, idx.shape[1]):
            j = idx[i, jpos]
            rows.append(i); cols.append(j); vals.append(w_local[i, jpos])
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = np.array(vals, dtype=np.float32)
    W = sp.coo_matrix((vals, (rows, cols)), shape=(Nw, Nw))
    W = 0.5*(W + W.T)
    W = W.tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float32) + 1e-8
    r_list, c_list, v_list = [], [], []
    W = W.tocoo()
    for r, c, v in zip(W.row, W.col, W.data):
        v_norm = v / np.sqrt(deg[r] * deg[c])
        r_list.append(r); c_list.append(c); v_list.append(v_norm)
    return np.asarray(r_list, np.int64), np.asarray(c_list, np.int64), np.asarray(v_list, np.float32), deg

def precompute_subgraphs_diffusion(psi: np.ndarray, wins, C: int, cfg: Cfg):
    full_sub, center_sub, center_deg_list = [], [], []
    for (st, ed, cst, ced) in wins:
        psi_win = psi[st:ed]
        r, c, v, deg = _knn_on_diffusion(psi_win, k=cfg.knn_k, use_dist_w=cfg.use_distance_weight)
        full_sub.append((r, c, v))
        Bf = ed - st; Bc = ced - cst; c0 = min(C, Bf); c1 = c0 + Bc
        m = (r >= c0) & (r < c1) & (c >= c0) & (c < c1)
        if m.any():
            rc = r[m] - c0; cc = c[m] - c0; vv = v[m]
        else:
            rc = np.empty(0, np.int64); cc = np.empty(0, np.int64); vv = np.empty(0, np.float32)
        center_sub.append((rc, cc, vv))
        center_deg_list.append(deg[c0:c1].copy() if Bc > 0 else np.array([], dtype=np.float32))
    return full_sub, center_sub, center_deg_list

def build_msd_features(adata, order, n_dims=20, scales=(1, 2, 4, 8)):
    if "diffmap_evals" not in adata.uns or "X_diffmap" not in adata.obsm:
        sc.tl.diffmap(adata)
    psi = np.asarray(adata.obsm["X_diffmap"])[order, :n_dims]
    evals = np.asarray(adata.uns["diffmap_evals"])[:n_dims]
    evals = np.clip(evals, 1e-6, 1.0)
    np.nan_to_num(psi, copy=False); np.nan_to_num(evals, copy=False)
    feats = [(psi * (evals ** t)[None, :]) for t in scales]
    X_msd = np.concatenate(feats, axis=1).astype(np.float32)
    X_msd -= X_msd.mean(0, keepdims=True)
    X_msd /= (X_msd.std(0, keepdims=True) + 1e-8)
    return np.nan_to_num(X_msd, copy=False)

def prepare_pseudotime_and_graph(adata):
    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata); sanitize_adata(adata)
    try:
        if "stage" in adata.obs:
            sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", batch_key="stage")
            adata._inplace_subset_var(adata.var["highly_variable"].values)
    except Exception:
        pass
    sc.tl.pca(adata, n_comps=50, svd_solver="randomized")
    Xp = adata.obsm.get("X_pca", None); np.nan_to_num(Xp, copy=False)
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=min(50, Xp.shape[1]))
    sc.tl.diffmap(adata)
    if "X_diffmap" in adata.obsm: np.nan_to_num(adata.obsm["X_diffmap"], copy=False)
    G = adata.obsp["connectivities"].tocsr()
    root_idx = None
    if "stage" in adata.obs:
        s = adata.obs["stage"].astype("category")
        if len(s.cat.categories) > 0:
            earliest = s.cat.categories[0]
            idx = np.where(adata.obs["stage"].values == earliest)[0]
            if idx.size > 0:
                deg = np.asarray(G[idx][:, :].sum(axis=1)).ravel()
                root_idx = int(idx[int(deg.argmax())])
    if root_idx is None:
        DC = adata.obsm["X_diffmap"]; root_idx = int(np.argmin(DC[:, 0]))
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    order = np.argsort(np.asarray(adata.obs["dpt_pseudotime"]))
    G_sorted = G[order][:, order].tocoo()
    return order, G_sorted


# ##############################################################################
#  TRAINING
# ##############################################################################

def train_overlap_graph_one_stage(model, X_full, X_hd, psi_sorted, cfg: Cfg, device, wins,
                                  full_sub, center_sub, center_deg_list, P_computer,
                                  layer_name: Optional[str] = None,
                                  stage_codes_sorted: Optional[np.ndarray] = None):
    model.to(device)
    actual_lr = cfg.learning_rate
    if layer_name == "Dense3":
        actual_lr = cfg.learning_rate * 0.3
        print(f"[Dense3] Using gentle LR = {actual_lr:.6f}")
    opt = AdamW(model.parameters(), lr=actual_lr, weight_decay=cfg.weight_decay)
    T_0 = 20 if cfg.early_exag_epochs > 0 else 10
    sch = CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=2)
    KL = create_kl_divergence_stable_weighted(cfg.low_dim, alpha_override=cfg.kl_alpha)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and torch.cuda.is_available()))
    X_t = torch.as_tensor(X_full, dtype=torch.float32, device=device)
    B = int(cfg.batch_size); C = int(cfg.context)

    P_cache_stage1 = {}
    if layer_name is None:
        print(f"[{layer_name or 'Stage1'}] Precomputing P matrices with backend: {cfg.P_backend}")
        t0 = time.time()
        for widx, (st, ed, cst, ced) in enumerate(wins):
            Bc = ced - cst
            if Bc <= 1: continue
            Xc = X_hd[cst:ced]
            deg_c = center_deg_list[widx] if widx < len(center_deg_list) else None
            cache_key = f"stage1_w{widx}" if cfg.P_backend == "cached_knn" else None
            P_cache_stage1[(cst, Bc)] = P_computer.compute_P_mixture(Xc, cfg.mix_perplexities, deg_c, cache_key)
        print(f"[{layer_name or 'Stage1'}] P computation took {time.time()-t0:.2f}s")

    P_list_dense = None
    best, best_state, bad = float("inf"), None, 0

    for ep in range(cfg.max_epochs):
        model.train(); tot, steps = 0.0, 0
        refresh_every = cfg.refresh_P_every_epochs_warm if ep < cfg.warmup_epochs else cfg.refresh_P_every_epochs_late

        if layer_name is not None and ((ep % max(1, refresh_every) == 0) or P_list_dense is None):
            P_list_dense = []
            for widx, (st, ed, cst, ced) in enumerate(wins):
                Bf = ed - st; Bc = ced - cst; c0 = min(C, Bf)
                if Bc <= 1:
                    P_list_dense.append(np.eye(max(Bc,1), dtype=np.float32)/max(Bc,1)); continue
                X_blk = X_t[st:ed]
                with torch.no_grad():
                    layer_map = {"Dense1": 1, "Dense2": 2, "Dense3": 3}
                    feats_full = model.get_layer_output(X_blk, layer_map[layer_name]).cpu().numpy()
                Xc = feats_full[c0:c0+Bc]
                deg_c = center_deg_list[widx] if widx < len(center_deg_list) else None
                cache_key = f"{layer_name}_e{ep}_w{widx}" if cfg.P_backend == "cached_knn" else None
                P_list_dense.append(P_computer.compute_P_mixture(Xc, cfg.mix_perplexities, deg_c, cache_key))
            if cfg.P_backend == "cached_knn" and ep > 0:
                old_prefix = f"{layer_name}_e{ep-refresh_every}"
                for k in [k for k in list(P_computer.knn_cache.keys()) if k.startswith(old_prefix)]:
                    del P_computer.knn_cache[k]

        for widx, (st, ed, cst, ced) in enumerate(wins):
            Bf = ed - st; Bc = ced - cst
            if Bc <= 1: continue
            X_blk = X_t[st:ed]
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and torch.cuda.is_available()):
                Y_full = model(X_blk); c0 = min(C, Bf); Y_center = Y_full[c0:c0+Bc]
            P_np = P_cache_stage1.get((cst, Bc)) if layer_name is None else (P_list_dense[widx] if widx < len(P_list_dense) else None)
            if P_np is None: continue
            P_t = torch.as_tensor(P_np, dtype=torch.float32, device=device)

            # Early exaggeration: multiply P and linearly decay to 1.0.
            if cfg.early_exag_factor > 1.0 and ep < cfg.early_exag_epochs:
                exag = cfg.early_exag_factor * (1.0 - ep / cfg.early_exag_epochs) + 1.0 * (ep / cfg.early_exag_epochs)
                P_t = P_t * exag
                P_t = P_t / (P_t.sum() + 1e-15)

            deg_c = center_deg_list[widx] if widx < len(center_deg_list) else None
            if deg_c is not None and deg_c.size == Bc:
                row_w_np = 1.0 / (deg_c.astype(np.float32) + 1e-6)
                row_w_np /= (row_w_np.mean() + 1e-8)
                row_w_t = torch.as_tensor(row_w_np, dtype=torch.float32, device=device)
            else:
                row_w_t = None

            if cfg.graph_center_only:
                r_np, c_np, v_np = center_sub[widx]
                if r_np.size == 0:
                    L_g = Y_full.new_tensor(0.0)
                else:
                    r = torch.as_tensor(r_np, dtype=torch.long, device=device)
                    c = torch.as_tensor(c_np, dtype=torch.long, device=device)
                    v = torch.as_tensor(v_np, dtype=torch.float32, device=device)
                    with torch.amp.autocast("cuda", enabled=cfg.use_amp and torch.cuda.is_available()):
                        L_g = graph_smoothness_loss(Y_center, r, c, v)
            else:
                r_np, c_np, v_np = full_sub[widx]
                if r_np.size == 0:
                    L_g = Y_full.new_tensor(0.0)
                else:
                    r = torch.as_tensor(r_np, dtype=torch.long, device=device)
                    c = torch.as_tensor(c_np, dtype=torch.long, device=device)
                    v = torch.as_tensor(v_np, dtype=torch.float32, device=device)
                    with torch.amp.autocast("cuda", enabled=cfg.use_amp and torch.cuda.is_available()):
                        L_g = graph_smoothness_loss(Y_full, r, c, v)

            with torch.amp.autocast("cuda", enabled=cfg.use_amp and torch.cuda.is_available()):
                L_kl = KL(P_t, Y_center, row_weights=row_w_t)
                loss = L_kl + cfg.lambda_graph * L_g
                # Contrastive loss with configurable stages.
                has_contr = stage_codes_sorted is not None and cfg.lambda_contrastive > 0 and layer_name is not None
                if has_contr:
                    has_contr = layer_name in cfg.contrastive_stages
                if has_contr:
                    codes_center = torch.as_tensor(stage_codes_sorted[cst:ced], dtype=torch.long, device=device)
                    n_contr = min(cfg.contrastive_n_samples, max(64, Bc // 2))
                    L_contr = contrastive_trajectory_loss(Y_center, codes_center,
                                                         margin=cfg.contrastive_margin, n_samples=n_contr)
                    loss = loss + cfg.lambda_contrastive * L_contr
                # Centroid ordering loss.
                if cfg.lambda_ordering > 0 and stage_codes_sorted is not None and layer_name in ("Dense2", "Dense3"):
                    codes_ctr = torch.as_tensor(stage_codes_sorted[cst:ced], dtype=torch.long, device=device)
                    n_stg = int(codes_ctr.max().item()) + 1
                    L_ord = centroid_ordering_loss(Y_center, codes_ctr, n_stg)
                    loss = loss + cfg.lambda_ordering * L_ord
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt); scaler.update()
            tot += float(loss.item()); steps += 1

        if steps == 0:
            print(f"[Warn] epoch {ep+1}: no valid steps"); continue
        sch.step()
        avg = tot/steps
        print(f"[{layer_name or 'Stage1'}] Epoch {ep+1}/{cfg.max_epochs} | loss={avg:.6f}")
        if avg < best - 1e-6:
            best, bad = avg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[EarlyStop] best={best:.6f}"); break

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        Y = model(torch.as_tensor(X_full, dtype=torch.float32, device=device)).float().cpu().numpy()
    return model, Y

def do_multi_stage_dr_training_graph(model, X_full, X_hd, psi_sorted, cfg: Cfg, device, wins,
                                     full_sub, center_sub, center_deg_list,
                                     stage_codes_sorted: Optional[np.ndarray] = None):
    P_computer = FastPComputer(cfg, device)
    outputs = []
    print("\n" + "="*50); print(f"Starting Stage 1 with P backend: {cfg.P_backend}"); print("="*50)
    if stage_codes_sorted is not None and cfg.lambda_contrastive > 0:
        print(f"[Strategy C] Contrastive trajectory loss (lambda={cfg.lambda_contrastive}) will be applied in Dense3")
    model, Y1 = train_overlap_graph_one_stage(model, X_full, X_hd, psi_sorted, cfg, device, wins,
                                              full_sub, center_sub, center_deg_list, P_computer,
                                              layer_name=None, stage_codes_sorted=stage_codes_sorted)
    outputs.append(("stage1", Y1))
    stage_names = ["Dense1", "Dense2", "Dense3"]
    for i in range(1, min(cfg.n_recur, 4)):
        print("\n" + "="*50); print(f"Starting Stage {i+1}: {stage_names[i-1]}"); print("="*50)
        model, Y = train_overlap_graph_one_stage(model, X_full, X_hd, psi_sorted, cfg, device, wins,
                                                 full_sub, center_sub, center_deg_list, P_computer,
                                                 layer_name=stage_names[i-1],
                                                 stage_codes_sorted=stage_codes_sorted)
        outputs.append((stage_names[i-1].lower(), Y))
    P_computer.clear_cache()
    return model, outputs

def benchmark_P_backends(X_sample: np.ndarray, cfg: Cfg, device):
    print("\n" + "="*50); print("Benchmarking P computation backends"); print(f"Data shape: {X_sample.shape}"); print("="*50)
    backends = ["cached_knn", "torch_exact"]
    if X_sample.shape[0] <= 500: backends.insert(0, "original")
    results = {}
    for backend in backends:
        cfg_test = Cfg(); cfg_test.P_backend = backend
        P_computer = FastPComputer(cfg_test, device)
        print(f"Testing {backend}...", end=" ")
        t0 = time.time()
        try:
            P = P_computer.compute_P_mixture(X_sample, cfg.mix_perplexities, None, "benchmark")
            elapsed = time.time() - t0; results[backend] = {"time": elapsed, "ok": True}
            print(f"{elapsed:.2f}s | sum={P.sum():.6f}")
        except Exception as e:
            results[backend] = {"time": None, "ok": False, "err": str(e)}
            print(f"FAILED - {e}")
    ok = {k: v for k, v in results.items() if v["ok"]}
    if ok:
        fastest = min(ok.keys(), key=lambda k: ok[k]["time"])
        print(f"\nRecommended backend: {fastest} ({ok[fastest]['time']:.2f}s)")
        return fastest
    return "cached_knn"


# ##############################################################################
#  BASELINES (from v43)
# ##############################################################################

def run_baselines(adata, X_pca, dc_off=1):
    results = {}
    for name, fn in [
        ("PCA", lambda: PCA(n_components=2, random_state=0).fit_transform(X_pca)),
        ("t-SNE", lambda: TSNE(n_components=2, perplexity=30, random_state=0,
                               init="pca", learning_rate="auto").fit_transform(X_pca)),
        ("UMAP", lambda: umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                                    random_state=0).fit_transform(X_pca) if HAS_UMAP else None),
        ("PHATE", lambda: phate.PHATE(n_components=2, random_state=0, n_jobs=-1,
                                       verbose=0).fit_transform(X_pca) if HAS_PHATE else None),
        ("DiffMap", lambda: np.asarray(adata.obsm.get("X_diffmap", np.zeros((adata.n_obs, 4))))[:, dc_off:dc_off+2].copy()),
    ]:
        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            Y = fn(); elapsed = time.time()-t0
            if Y is not None: results[name] = (Y, elapsed); print(f"done ({elapsed:.1f}s)")
            else: print("skipped")
        except Exception as e: print(f"failed ({e})")
    return results


# ##############################################################################
#  EVALUATION METRICS (from v43, lines ~1400-1610 -- copied exactly)
# ##############################################################################

def _knn_idx(X, k=15):
    k = min(k, max(2, X.shape[0]-1))
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, idx = nbrs.kneighbors(X)
    return idx[:, 1:]

def _stratified_subsample(codes, max_n, rng):
    unique = np.unique(codes); n = len(codes); indices = []
    for c in unique:
        c_idx = np.where(codes == c)[0]
        n_take = max(1, min(len(c_idx), int(round(len(c_idx)/n * max_n))))
        indices.append(rng.choice(c_idx, n_take, replace=False))
    return np.sort(np.concatenate(indices))

def _tsp_order(centroids):
    """Shortest Hamiltonian path via Held-Karp DP. Feasible for n<=20."""
    n = len(centroids)
    dist = np.sqrt(((centroids[:, None] - centroids[None, :]) ** 2).sum(-1))
    INF = 1e18
    dp = np.full((1 << n, n), INF)
    parent = np.full((1 << n, n), -1, dtype=int)
    for i in range(n):
        dp[1 << i, i] = 0.0
    for mask in range(1, 1 << n):
        for u in range(n):
            if not (mask & (1 << u)): continue
            if dp[mask, u] >= INF: continue
            for v in range(n):
                if mask & (1 << v): continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask, u] + dist[u, v]
                if new_cost < dp[new_mask, v]:
                    dp[new_mask, v] = new_cost
                    parent[new_mask, v] = u
    full_mask = (1 << n) - 1
    best_end = int(np.argmin(dp[full_mask]))
    path = []
    mask, u = full_mask, best_end
    while u >= 0:
        path.append(u)
        prev = parent[mask, u]
        mask ^= (1 << u)
        u = prev
    path.reverse()
    return path

def _polyline_arc_project(Y, centroids):
    """Project each point onto the polyline through centroids, return arc-lengths."""
    n_seg = len(centroids) - 1
    seg_vecs = centroids[1:] - centroids[:-1]
    seg_lens_sq = (seg_vecs ** 2).sum(1)
    seg_lens = np.sqrt(seg_lens_sq)
    cum_lens = np.concatenate([[0], np.cumsum(seg_lens)])
    best_dist = np.full(len(Y), np.inf)
    best_arc = np.zeros(len(Y))
    for s in range(n_seg):
        v = Y - centroids[s]
        if seg_lens_sq[s] < 1e-20:
            t = np.zeros(len(Y))
        else:
            t = (v @ seg_vecs[s]) / seg_lens_sq[s]
            t = np.clip(t, 0, 1)
        proj = centroids[s] + t[:, None] * seg_vecs[s]
        d = np.sqrt(((Y - proj) ** 2).sum(1))
        mask = d < best_dist
        best_dist[mask] = d[mask]
        best_arc[mask] = cum_lens[s] + t[mask] * seg_lens[s]
    return best_arc

# ----- Tier 1: Essential for cell differentiation -----

def metric_stage_order(Y, codes, n_stages):
    """Stage ordering via arc-length projection onto centroid polyline.
    Curve-aware: follows the trajectory curve instead of projecting to a line.
    Uses ALL cells (not just centroids) for stability."""
    centroids = []
    for s in range(n_stages):
        m = codes == s
        if m.sum() > 0: centroids.append(Y[m].mean(0))
    if len(centroids) < 2: return np.nan
    centroids = np.array(centroids)
    arc = _polyline_arc_project(Y, centroids)
    return float(abs(spearmanr(codes, arc, nan_policy="omit").statistic))

def metric_ordered_stage_separability(Y, codes, n_stages):
    """
    Combined metric: ordering quality (arc-length) * visual separation (Fisher ratio).
    Arc-length projection follows the trajectory curve — no PCA linearity assumption.
    """
    centroids, within_stds = [], []
    valid_idx = []
    for s in range(n_stages):
        m = codes == s
        if m.sum() < 3: continue
        c = Y[m].mean(0)
        centroids.append(c)
        within_stds.append(np.sqrt(np.mean(np.sum((Y[m] - c) ** 2, axis=1))) + 1e-8)
        valid_idx.append(s)

    if len(centroids) < 3: return np.nan
    centroids = np.array(centroids)

    # Arc-length projection for curve-aware ordering
    arc = _polyline_arc_project(Y, centroids)
    # Compute mean arc-length per stage for ordering check
    stage_arcs = []
    for k in range(len(valid_idx)):
        m = codes == valid_idx[k]
        stage_arcs.append(np.mean(arc[m]))

    scores = []
    for k in range(len(valid_idx) - 1):
        dist = np.linalg.norm(centroids[k+1] - centroids[k])
        avg_within = 0.5 * (within_stds[k] + within_stds[k+1])
        fisher = dist / avg_within
        # Ordering via arc-length: does mean arc increase with stage?
        ordered = 1.0 if stage_arcs[k+1] > stage_arcs[k] else 0.0
        scores.append(ordered * float(np.tanh(fisher)))

    return float(np.mean(scores)) if scores else np.nan

def metric_spearman_rho(Y, dpt, codes=None, n_stages=None, seed=0):
    """Global pseudotime monotonicity via trajectory arc-length projection."""
    try:
        # Build centroid polyline for curve-aware projection
        if codes is not None and n_stages is not None:
            centroids = []
            for s in range(n_stages):
                m = codes == s
                if m.sum() > 0: centroids.append(Y[m].mean(0))
            if len(centroids) >= 2:
                centroids = np.array(centroids)
                arc = _polyline_arc_project(Y, centroids)
                return float(abs(spearmanr(dpt, arc, nan_policy="omit").statistic))
        # Fallback to PCA if no stage info
        pc1 = PCA(n_components=1, random_state=seed).fit_transform(Y)[:, 0]
        return float(abs(spearmanr(dpt, pc1, nan_policy="omit").statistic))
    except: return np.nan

def metric_stage_compactness(Y, codes, n_stages):
    """
    Mean ratio of within-stage spread to between-stage spread.
    Low = compact stages with large gaps (good for differentiation viz).
    """
    within_vars, between = [], []
    centroids = []
    for s in range(n_stages):
        m = codes == s
        if m.sum() < 3: continue
        c = Y[m].mean(0); centroids.append(c)
        within_vars.append(np.mean(np.sum((Y[m] - c) ** 2, axis=1)))
    if len(centroids) < 2: return np.nan
    centroids = np.array(centroids)
    global_c = centroids.mean(0)
    between_var = np.mean(np.sum((centroids - global_c) ** 2, axis=1))
    mean_within = np.mean(within_vars)
    # Compactness = between / (within + eps), higher = better separation
    return float(between_var / (mean_within + 1e-8))


# ----- Tier 2: Trajectory structure -----

def metric_dpt_knn_corr(Y, dpt, k=15):
    idx = _knn_idx(Y, k); corrs = []
    for i in range(Y.shape[0]):
        emb_d = np.sqrt(np.sum((Y[idx[i]] - Y[i])**2, axis=1))
        dpt_d = np.abs(dpt[idx[i]] - dpt[i])
        if np.std(emb_d) > 1e-8 and np.std(dpt_d) > 1e-8:
            r = spearmanr(emb_d, dpt_d).statistic
            if not np.isnan(r): corrs.append(r)
    return float(np.mean(corrs)) if corrs else np.nan

def metric_gradient_smooth(Y, dpt, k=15):
    idx = _knn_idx(Y, k); dpt_std = np.std(dpt)
    if dpt_std < 1e-8: return np.nan
    devs = [abs(dpt[i] - np.mean(dpt[idx[i]])) / dpt_std for i in range(Y.shape[0])]
    return float(1.0 - min(np.mean(devs), 1.0))

def metric_temporal_coherence(Y, dpt, k=15):
    idx = _knn_idx(Y, k); dpt_range = np.nanmax(dpt) - np.nanmin(dpt)
    if dpt_range < 1e-8: return np.nan
    variances = [np.var(dpt[idx[i]]) / (dpt_range**2) for i in range(Y.shape[0])]
    return float(1.0 - min(np.mean(variances) * 10, 1.0))

def metric_trajectory_continuity(Y, dpt, n_bins=50):
    """
    Measures how well the mean trajectory curve is continuous.
    Bins cells by pseudotime, computes centroid per bin, measures:
    1. Smoothness of centroid path (penalize sharp turns)
    2. Correlation of arc length with pseudotime
    Returns geometric mean of both.
    """
    valid = ~np.isnan(dpt)
    if valid.sum() < 20: return np.nan
    Y_v, dpt_v = Y[valid], dpt[valid]
    bins = np.linspace(dpt_v.min(), dpt_v.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(dpt_v, bins) - 1, 0, n_bins - 1)

    cx, cy, bt = [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            cx.append(Y_v[m, 0].mean()); cy.append(Y_v[m, 1].mean()); bt.append(dpt_v[m].mean())
    if len(bt) < 5: return np.nan
    cx, cy, bt = np.array(cx), np.array(cy), np.array(bt)

    # Arc length vs pseudotime correlation
    steps = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
    arc = np.concatenate([[0], np.cumsum(steps)])
    if np.std(arc) < 1e-8 or np.std(bt) < 1e-8: return np.nan
    r_arc = abs(spearmanr(arc, bt).statistic)

    # Smoothness: penalize sharp direction changes
    if len(cx) < 4: return float(r_arc)
    dx = np.diff(cx); dy = np.diff(cy)
    norms = np.sqrt(dx**2 + dy**2) + 1e-10
    dx_n = dx / norms; dy_n = dy / norms
    # Cosine of angle between consecutive segments
    cos_angles = dx_n[:-1]*dx_n[1:] + dy_n[:-1]*dy_n[1:]
    cos_angles = np.clip(cos_angles, -1, 1)
    smoothness = float(np.mean((cos_angles + 1) / 2))  # 0=U-turn, 1=straight

    return float(np.sqrt(r_arc * smoothness))  # geometric mean

def metric_boundary_mixing(Y, codes, k=15):
    idx = _knn_idx(Y, k); n_stages = int(codes.max()) + 1; scores = []
    for s in range(n_stages - 1):
        cells = np.where((codes == s) | (codes == s + 1))[0]
        if len(cells) < 10: continue
        cross = []
        for i in cells:
            if codes[i] == s: cross.append(np.mean(codes[idx[i]] == s+1))
            else: cross.append(np.mean(codes[idx[i]] == s))
        if cross: scores.append(np.mean(cross))
    return float(np.mean(scores)) if scores else np.nan


# ----- Tier 3: Structural fidelity -----

def metric_trustworthiness(X_hd, Y, k=15):
    k_tw = min(k, max(2, X_hd.shape[0] - 2))
    return float(trustworthiness(X_hd, Y, n_neighbors=k_tw))

def metric_continuity_jaccard(X_hd, Y, k=15):
    idx_hd = _knn_idx(X_hd, k); idx_ld = _knn_idx(Y, k)
    cont = [len(set(idx_hd[i]) & set(idx_ld[i])) / k for i in range(Y.shape[0])]
    jacc = [len(set(idx_hd[i]) & set(idx_ld[i])) / len(set(idx_hd[i]) | set(idx_ld[i])) for i in range(Y.shape[0])]
    return float(np.mean(cont)), float(np.mean(jacc))


def metric_ddc(Y, dpt, k=15):
    """Developmental Direction Consistency: checks if later-pseudotime neighbors
    lie 'ahead' and earlier ones lie 'behind' each cell in embedding space."""
    valid = ~np.isnan(dpt)
    if valid.sum() < 50: return np.nan
    Y_v, dpt_v = Y[valid], dpt[valid]
    idx = _knn_idx(Y_v, k)
    scores = []
    for i in range(len(Y_v)):
        nn = idx[i]
        dt = dpt_v[nn] - dpt_v[i]
        ahead = nn[dt > 0]; behind = nn[dt < 0]
        if len(ahead) == 0 or len(behind) == 0:
            continue
        dir_ahead = (Y_v[ahead] - Y_v[i]).mean(0)
        dir_behind = (Y_v[behind] - Y_v[i]).mean(0)
        na = np.linalg.norm(dir_ahead); nb = np.linalg.norm(dir_behind)
        if na < 1e-10 or nb < 1e-10:
            continue
        # Cosine between ahead direction and negative-behind direction
        cos = np.dot(dir_ahead, -dir_behind) / (na * nb)
        scores.append((cos + 1) / 2)  # Normalize to [0, 1]
    return float(np.mean(scores)) if scores else np.nan

def metric_pgs(Y, dpt, k=15):
    """Pseudotime Gradient Smoothness: measures how smoothly the pseudotime
    gradient direction changes among neighboring cells."""
    valid = ~np.isnan(dpt)
    if valid.sum() < 50: return np.nan
    Y_v, dpt_v = Y[valid], dpt[valid]
    idx = _knn_idx(Y_v, k)
    # Compute per-cell pseudotime gradient in embedding space
    grads = np.zeros_like(Y_v)
    for i in range(len(Y_v)):
        nn = idx[i]
        dy = Y_v[nn] - Y_v[i]              # (k, 2) spatial displacement
        dt = dpt_v[nn] - dpt_v[i]           # (k,) pseudotime difference
        # Weighted gradient: sum of displacement * pseudotime_diff
        grads[i] = (dy * dt[:, None]).mean(0)
    # Compute norms and normalize
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    grads_n = grads / norms
    # For each cell, compute mean cosine similarity with neighbors' gradients
    scores = []
    for i in range(len(Y_v)):
        nn = idx[i]
        cos_sims = grads_n[nn] @ grads_n[i]  # (k,) cosine similarities
        scores.append(np.mean(cos_sims))
    return float((np.mean(scores) + 1) / 2)  # Normalize to [0, 1]


def evaluate_embedding(Y, X_hd, dpt, codes, k=15, seed=0, max_cells=10000):
    n = Y.shape[0]; rng = np.random.RandomState(seed); n_stages = int(codes.max()) + 1
    if n > max_cells:
        ix = _stratified_subsample(codes, max_cells, rng)
        Y, X_hd, dpt, codes = Y[ix], X_hd[ix], dpt[ix], codes[ix]
    m = {}

    # Tier 1: Cell differentiation essentials
    m["Stage_Order"] = metric_stage_order(Y, codes, n_stages)
    m["Ordered_Separability"] = metric_ordered_stage_separability(Y, codes, n_stages)
    m["Spearman_rho"] = metric_spearman_rho(Y, dpt, codes, n_stages, seed)
    m["Stage_Compactness"] = metric_stage_compactness(Y, codes, n_stages)

    # Tier 2: Trajectory structure
    m["DPT_kNN_Corr"] = metric_dpt_knn_corr(Y, dpt, k)
    m["Gradient_Smooth"] = metric_gradient_smooth(Y, dpt, k)
    m["Temporal_Coherence"] = metric_temporal_coherence(Y, dpt, k)
    m["Trajectory_Continuity"] = metric_trajectory_continuity(Y, dpt)
    m["Boundary_Mixing"] = metric_boundary_mixing(Y, codes, k)
    m["DDC"] = metric_ddc(Y, dpt, k)
    m["PGS"] = metric_pgs(Y, dpt, k)

    # Tier 3: Structural
    m["Trustworthiness"] = metric_trustworthiness(X_hd, Y, k)
    cont, jacc = metric_continuity_jaccard(X_hd, Y, k)
    m["Continuity"] = cont; m["kNN_Jaccard"] = jacc

    # Reference only
    u = np.unique(codes)
    m["Silhouette_ref"] = silhouette_score(Y, codes) if len(u) >= 2 else np.nan

    return m


# ##############################################################################
#  PLOTTING (Nature style, from v43)
# ##############################################################################

MM2IN = 1/25.4; SINGLE_COL = 89*MM2IN; DOUBLE_COL = 183*MM2IN

STAGE_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]
METHOD_COLORS = {"PCA": "#A0A0A0", "t-SNE": "#4DBBD5", "UMAP": "#E64B35",
    "PHATE": "#00A087", "DiffMap": "#F39B7F", "Bio-PD": "#3C5488"}
DPT_CMAP = LinearSegmentedColormap.from_list("dpt", ["#440154","#31688E","#35B779","#FDE725"], N=256)

def _mc(name):
    if name in METHOD_COLORS: return METHOD_COLORS[name]
    for k, v in METHOD_COLORS.items():
        if k.lower() in name.lower(): return v
    return "#3C5488"

def setup_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
        "lines.linewidth": 0.8, "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def plot_embedding_gallery(all_results, codes, dpt, stage_names, out_dir):
    """Gallery plot: all methods side-by-side with stage and pseudotime coloring."""
    methods = list(all_results.keys()); n = len(methods)
    fig, axes = plt.subplots(2, n, figsize=(DOUBLE_COL, DOUBLE_COL*0.38))
    if n == 1: axes = axes.reshape(2, 1)
    for j, mn in enumerate(methods):
        Y = all_results[mn][0] if isinstance(all_results[mn], tuple) else all_results[mn]
        for si, sn in enumerate(stage_names):
            mask = codes == si
            if mask.sum() == 0: continue
            axes[0, j].scatter(Y[mask, 0], Y[mask, 1], s=0.3, alpha=0.4,
                              c=STAGE_PALETTE[si % len(STAGE_PALETTE)], rasterized=True, lw=0)
        axes[0, j].set_title(mn, fontsize=7, pad=2)
        axes[1, j].scatter(Y[:, 0], Y[:, 1], s=0.3, c=dpt, cmap=DPT_CMAP, alpha=0.4, rasterized=True, lw=0)
        for ax in [axes[0, j], axes[1, j]]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
    axes[0, 0].set_ylabel("Stage", fontsize=7); axes[1, 0].set_ylabel("Pseudotime", fontsize=7)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STAGE_PALETTE[i % len(STAGE_PALETTE)],
                      markersize=3, label=sn, lw=0) for i, sn in enumerate(stage_names)]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(stage_names)), fontsize=5, frameon=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.16, top=0.94, left=0.04, right=0.97)
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(out_dir, f"Fig1_embedding_gallery{ext}"), dpi=300)
    plt.close(fig)


def plot_metric_bars(df, out_dir):
    """Two-panel bar chart: Tier1 (differentiation) + Tier2 (trajectory)."""
    tier1 = [("Stage_Order", "Stage\nOrder \u2191"),
             ("Ordered_Separability", "Ordered\nSeparab. \u2191"),
             ("Stage_Compactness", "Stage\nCompact. \u2191"),
             ("Spearman_rho", "Spearman \u03c1\n(Global) \u2191")]
    tier2 = [("Trajectory_Continuity", "Trajectory\nContinuity \u2191"),
             ("DPT_kNN_Corr", "DPT-kNN\nCorr \u2191"),
             ("Temporal_Coherence", "Temporal\nCoherence \u2191"),
             ("Gradient_Smooth", "Gradient\nSmooth \u2191")]

    for tier, tier_name, fname in [(tier1, "Cell Differentiation Quality", "Fig2a"),
                                    (tier2, "Trajectory Structure Quality", "Fig2b")]:
        avail = [(k, l) for k, l in tier if k in df.columns]
        nm = len(avail); nme = len(df)
        fig, axes = plt.subplots(1, nm, figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.35))
        if nm == 1: axes = [axes]
        bw = 0.72 / nme
        for mi, (mk, ml) in enumerate(avail):
            ax = axes[mi]
            for j, method in enumerate(df["Method"]):
                val = df.loc[df["Method"] == method, mk].values[0]
                if np.isnan(val): val = 0
                c = _mc(method); ec = "#222" if "(Ours)" in str(method) else "none"
                lw_e = 0.8 if "(Ours)" in str(method) else 0
                ax.bar(j * bw - (nme-1)*bw/2, val, bw*0.9, color=c, alpha=0.85, edgecolor=ec, linewidth=lw_e)
            ax.set_title(ml, fontsize=5.5, pad=3); ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=5)
            if mi == 0: ax.set_ylabel("Score", fontsize=6)
        lp = [Patch(facecolor=_mc(m), label=m) for m in df["Method"]]
        fig.legend(handles=lp, loc="lower center", ncol=min(7, nme), fontsize=5, frameon=False)
        fig.suptitle(tier_name, fontsize=8, fontweight="bold", y=0.98)
        fig.subplots_adjust(wspace=0.35, bottom=0.25, top=0.88)
        for ext in (".pdf", ".png"):
            fig.savefig(os.path.join(out_dir, f"{fname}_metrics{ext}"), dpi=300)
        plt.close(fig)


def plot_summary_table(df, out_dir):
    tier1 = ["Stage_Order", "Ordered_Separability", "Stage_Compactness", "Spearman_rho"]
    tier2 = ["DPT_kNN_Corr", "Gradient_Smooth", "Trajectory_Continuity", "Temporal_Coherence", "Boundary_Mixing"]
    tier3 = ["Trustworthiness", "Continuity", "kNN_Jaccard", "Silhouette_ref"]
    all_cols = ["Method", "Runtime_s"] + [c for c in tier1 + tier2 + tier3 if c in df.columns]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35)); ax.axis("off")
    df_d = df[all_cols].copy()
    for c in df_d.columns:
        if c == "Method": continue
        df_d[c] = df_d[c].apply(lambda x: f"{x:.3f}" if isinstance(x, float) and not np.isnan(x) else "-")
    table = ax.table(cellText=df_d.values,
                     colLabels=[c.replace("_", "\n") for c in df_d.columns],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(4.5); table.scale(1, 1.4)
    for j in range(len(all_cols)):
        cell = table[0, j]; cell.set_text_props(fontweight="bold", fontsize=4.5)
        if all_cols[j] in tier1: cell.set_facecolor("#D4E6F1")
        elif all_cols[j] in tier2: cell.set_facecolor("#D5F5E3")
        else: cell.set_facecolor("#E8E8E8")
    for j, c in enumerate(all_cols):
        if c in ("Method", "Runtime_s"): continue
        try:
            vals = pd.to_numeric(df[c], errors="coerce")
            best_idx = vals.idxmax()
            if not np.isnan(vals[best_idx]):
                table[best_idx + 1, j].set_text_props(fontweight="bold", color="#C0392B")
        except: pass
    fig.tight_layout(pad=0.2)
    for ext in (".pdf", ".png"): fig.savefig(os.path.join(out_dir, f"FigS_summary_table{ext}"), dpi=300)
    plt.close(fig)


def plot_radar(df, out_dir):
    """Radar chart comparing methods across metrics."""
    candidates = ["Stage_Order", "Ordered_Separability", "Stage_Compactness",
                   "Spearman_rho", "DPT_kNN_Corr", "Gradient_Smooth",
                   "Trajectory_Continuity", "Temporal_Coherence",
                   "Boundary_Mixing", "Trustworthiness", "Silhouette_ref"]
    candidates = [c for c in candidates if c in df.columns]
    biopd_idx = df.index[df["Method"].str.contains("Bio-PD", na=False)]
    if len(biopd_idx) == 0: return
    biopd_idx = biopd_idx[0]
    # Auto-select highlighted metrics based on method rank.
    selected = []
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().all(): continue
        rank = vals.rank(ascending=False, method="min").iloc[biopd_idx]
        if rank <= 3: selected.append((c, int(rank)))
    selected.sort(key=lambda x: x[1])
    cols = [s[0] for s in selected[:7]]
    if len(cols) < 4: cols = candidates[:5]
    name_map = {"Stage_Order": "Stage\nOrder", "Ordered_Separability": "Ordered\nSeparability",
                "Stage_Compactness": "Stage\nCompactness", "Spearman_rho": "Spearman rho",
                "DPT_kNN_Corr": "DPT-kNN\nCorrelation", "Gradient_Smooth": "Gradient\nSmoothness",
                "Trajectory_Continuity": "Trajectory\nContinuity", "Temporal_Coherence": "Temporal\nCoherence",
                "Boundary_Mixing": "Boundary\nMixing", "Trustworthiness": "Trust-\nworthiness",
                "Silhouette_ref": "Silhouette"}
    labels = [name_map.get(c, c.replace("_", "\n")) for c in cols]
    df_n = df[cols].copy()
    for c in cols:
        mn, mx = df_n[c].min(), df_n[c].max()
        df_n[c] = (df_n[c] - mn) / (mx - mn + 1e-8)
    nv = len(cols); angles = np.linspace(0, 2*np.pi, nv, endpoint=False).tolist() + [0]

    baseline_styles = {
        "PCA":     {"ls": (0, (1, 1)),     "marker": "s", "lw": 1.0, "ms": 3.5, "alpha": 0.85, "zorder": 3},
        "t-SNE":   {"ls": (0, (5, 2)),     "marker": "^", "lw": 1.0, "ms": 4.0, "alpha": 0.85, "zorder": 4},
        "UMAP":    {"ls": (0, (3, 1, 1, 1)), "marker": "D", "lw": 1.0, "ms": 3.5, "alpha": 0.85, "zorder": 5},
        "PHATE":   {"ls": (0, (5, 1)),     "marker": "v", "lw": 1.0, "ms": 4.0, "alpha": 0.85, "zorder": 6},
        "DiffMap": {"ls": (0, (3, 2)),     "marker": "P", "lw": 1.2, "ms": 4.5, "alpha": 0.90, "zorder": 7},
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL*1.8, SINGLE_COL*1.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5.5, color="#BBB")
    for r in [0.25, 0.50, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, lw=0.25, color="#DDD", zorder=0)
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1.12], lw=0.25, color="#DDD", zorder=0)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    method_order = [m for m in df["Method"] if "(Ours)" not in str(m)]
    method_order += [m for m in df["Method"] if "(Ours)" in str(m)]

    for m in method_order:
        i = df.index[df["Method"] == m][0]
        vals = df_n.iloc[i].tolist() + [df_n.iloc[i].tolist()[0]]
        color = _mc(m)
        is_ours = "(Ours)" in str(m)

        if is_ours:
            ax.plot(angles, vals, lw=2.8, label=m, color=color, zorder=10, ls="-",
                    marker="o", markersize=5.5, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=1.0)
            ax.fill(angles, vals, alpha=0.12, color=color, zorder=8)
        else:
            short = m.split()[0] if " " not in m else m
            for k in baseline_styles:
                if k in str(m): short = k; break
            sty = baseline_styles.get(short, {"ls": "--", "marker": "o", "lw": 0.9, "ms": 3, "alpha": 0.7, "zorder": 2})
            ax.plot(angles, vals, lw=sty["lw"], label=m, color=color,
                    zorder=sty["zorder"], ls=sty["ls"], alpha=sty["alpha"],
                    marker=sty["marker"], markersize=sty["ms"],
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7, fontweight="medium", color="#222")
    for lab, angle in zip(ax.get_xticklabels(), angles[:-1]):
        lab.set_y(lab.get_position()[1] - 0.02)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(-0.25, 1.22), fontsize=6,
                       frameon=True, framealpha=0.95, edgecolor="#DDD", fancybox=False,
                       handlelength=2.0, handletextpad=0.6, labelspacing=0.6,
                       borderpad=0.8, ncol=2, columnspacing=1.2)
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=1.5)
    for ext in (".pdf", ".png"): fig.savefig(os.path.join(out_dir, f"Fig3_radar{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage_progression_biopd(biopd_stage_results, codes, dpt, stage_names, out_dir):
    """
    Per-stage visualization: DiffMap + Stage1 + Dense1 + Dense2 + Dense3 side-by-side,
    with two rows (stage colors and pseudotime colors).
    """
    methods = list(biopd_stage_results.keys()); n = len(methods)
    if n == 0: return
    fig, axes = plt.subplots(2, n, figsize=(DOUBLE_COL, DOUBLE_COL*0.38))
    if n == 1: axes = axes.reshape(2, 1)
    n_stages = int(codes.max()) + 1
    for j, mn in enumerate(methods):
        Y = biopd_stage_results[mn]
        # Row 0: stage colors
        for si in range(min(n_stages, len(stage_names))):
            mask = codes == si
            if mask.sum() == 0: continue
            axes[0, j].scatter(Y[mask, 0], Y[mask, 1], s=0.3, alpha=0.4,
                              c=STAGE_PALETTE[si % len(STAGE_PALETTE)], rasterized=True, lw=0)
        axes[0, j].set_title(mn, fontsize=7, pad=2)
        # Row 1: pseudotime colors
        axes[1, j].scatter(Y[:, 0], Y[:, 1], s=0.3, c=dpt, cmap=DPT_CMAP, alpha=0.4, rasterized=True, lw=0)
        for ax in [axes[0, j], axes[1, j]]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
    axes[0, 0].set_ylabel("Stage", fontsize=7); axes[1, 0].set_ylabel("Pseudotime", fontsize=7)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=STAGE_PALETTE[i % len(STAGE_PALETTE)],
                      markersize=3, label=stage_names[i] if i < len(stage_names) else f"S{i}", lw=0)
               for i in range(min(n_stages, len(stage_names)))]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(stage_names)), fontsize=5, frameon=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.16, top=0.94, left=0.04, right=0.97)
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(out_dir, f"stage_progression{ext}"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Bio-PD public aliases
# ------------------------------------------------------------------------------
# Internal legacy symbol names are retained where they are part of the original
# computational implementation. Public-facing outputs and the command-line
# workflow use Bio-PD terminology.

# ======================================================================
#  PSEUDOTIME-ORDERED SUBSAMPLING
# ======================================================================

def stratified_subsample(stage_codes, dpt_norm, K=8000, seed=42):
    """Subsample cells proportionally by stage and uniformly along pseudotime."""
    rng = np.random.RandomState(seed)
    N = len(stage_codes)
    unique_stages = np.unique(stage_codes)
    n_stages = len(unique_stages)

    stage_sizes = np.array([np.sum(stage_codes == s) for s in unique_stages])
    stage_fracs = stage_sizes / stage_sizes.sum()
    per_stage = np.maximum(np.round(stage_fracs * K).astype(int), 10)

    while per_stage.sum() > K:
        per_stage[np.argmax(per_stage)] -= 1
    while per_stage.sum() < K:
        per_stage[np.argmin(per_stage / np.maximum(stage_sizes, 1))] += 1

    selected = []
    for si, s in enumerate(unique_stages):
        stage_mask = np.where(stage_codes == s)[0]
        n_want = min(per_stage[si], len(stage_mask))
        dpt_stage = dpt_norm[stage_mask]
        sorted_idx = stage_mask[np.argsort(dpt_stage)]
        if n_want >= len(sorted_idx):
            selected.extend(sorted_idx.tolist())
        else:
            step = len(sorted_idx) / n_want
            picks = [int(i * step) for i in range(n_want)]
            selected.extend(sorted_idx[picks].tolist())

    selected = np.array(sorted(set(selected)), dtype=np.int64)
    print(f"[SUBSAMPLE] {N} → {len(selected)} cells ({len(selected)/N*100:.1f}%), {n_stages} stages")
    return selected


def make_cfg_label_free(cfg):
    """Disable all stage-annotation-dependent loss weights in cfg."""
    cfg.lambda_contrastive = 0.0
    cfg.lambda_ordering = 0.0
    cfg.contrastive_stages = ()  # no stages get contrastive loss
    print(f"[CONFIG] Disabled stage-annotation-dependent losses: lambda_contrastive={cfg.lambda_contrastive}, "
          f"lambda_ordering={cfg.lambda_ordering}, contrastive_stages={cfg.contrastive_stages}")
    return cfg


# ======================================================================
#  BIO-PD CELL-LEVEL TRAINING PIPELINE
# ======================================================================

def train_biopd_cell_small(model, X_msd, psi_sorted, cfg, device, wins,
                        full_sub, center_sub, center_deg_list,
                        stage_codes_sorted, dpt_sorted):
    """Small-data execution path using the configured label-free objectives.

    Stage-annotation-dependent loss terms are disabled through cfg before all
    training stages. The original staged optimization sequence is retained.
    """
    Pc = FastPComputer(cfg, device)
    outputs = []

    # Ensure stage-annotation-dependent losses remain disabled.
    make_cfg_label_free(cfg)

    print(f"\n{'='*50}\nStage1: global objective with stage-dependent losses disabled\n{'='*50}")

    # Use the same overlap-graph training path for all stages with disabled stage-dependent weights.

    print(f"\n{'='*50}\nStage1: overlap-graph optimization\n{'='*50}")
    model, Y1 = train_overlap_graph_one_stage(
        model, X_msd, X_msd, psi_sorted, cfg, device, wins,
        full_sub, center_sub, center_deg_list, Pc,
        layer_name=None, stage_codes_sorted=stage_codes_sorted)
    outputs.append(("stage1", Y1))

    for nm in ["Dense1", "Dense2", "Dense3"]:
        print(f"\n{'='*50}\n{nm}: overlap-graph optimization with stage-dependent losses disabled\n{'='*50}")
        model, Y = train_overlap_graph_one_stage(
            model, X_msd, X_msd, psi_sorted, cfg, device, wins,
            full_sub, center_sub, center_deg_list, Pc,
            layer_name=nm, stage_codes_sorted=stage_codes_sorted)
        outputs.append((nm.lower(), Y))

    Pc.clear_cache()
    return model, outputs


def train_biopd_cell_large(adata, X_msd_full, psi_sorted_full, order,
                        stage_codes, dpt_norm, cfg, device,
                        subset_K=8000, seed=42,
                        skip_alignment=False,
                        override_lambda_graph=None,
                        override_lambda_flow=None,
                        override_lambda_dpt_smooth=None):
    """Large-data execution path with subset training, full inference, and alignment."""
    N = len(stage_codes)

    # Step 1: Subsample
    sub_idx = stratified_subsample(stage_codes, dpt_norm, K=subset_K, seed=seed)
    K_actual = len(sub_idx)

    # Step 2: Prepare subset data in DPT-sorted order
    dpt_sub = dpt_norm[sub_idx]
    sub_order = np.argsort(dpt_sub)
    sub_sorted_idx = sub_idx[sub_order]

    # Map to full sorted positions
    orig_to_sorted = np.zeros(N, dtype=np.int64)
    orig_to_sorted[order] = np.arange(N)
    sub_sorted_positions = orig_to_sorted[sub_sorted_idx]

    X_msd_sub = X_msd_full[sub_sorted_positions].copy().astype(np.float32)
    psi_sub = psi_sorted_full[sub_sorted_positions].copy()
    stage_codes_sub = stage_codes[sub_sorted_idx].astype(np.int64)
    dpt_sub_sorted = dpt_norm[sub_sorted_idx].astype(np.float32)

    # Build adjacency for subset
    try:
        adj_full = adata.obsp["connectivities"].tocsr()[order][:, order]
        adj_sub = adj_full[sub_sorted_positions][:, sub_sorted_positions].tocsr()
    except:
        adj_sub = None

    print(f"[SUBSET] K={K_actual}, MSD={X_msd_sub.shape}")

    # Step 3: Configure as small data
    sub_cfg = Cfg()
    sub_cfg.seed = seed
    sub_cfg.cuda_device = cfg.cuda_device
    sub_cfg.use_amp = True
    sub_cfg.P_backend = "cached_knn"
    torch.manual_seed(seed); np.random.seed(seed)
    adapt_cfg_to_dataset_size(sub_cfg, K_actual)
    make_cfg_label_free(sub_cfg)  # Disable stage-annotation-dependent losses.

    B, C = int(sub_cfg.batch_size), int(sub_cfg.context)
    S = B if (sub_cfg.center_nonoverlap or sub_cfg.stride is None) else int(sub_cfg.stride)
    wins = build_windows(K_actual, B, C, S)
    full_sub_g, center_sub_g, cdeg = precompute_subgraphs_diffusion(psi_sub, wins, C, sub_cfg)

    if sub_cfg.benchmark_on_start and K_actual > 100:
        sub_cfg.P_backend = benchmark_P_backends(
            X_msd_sub[np.random.choice(K_actual, min(500, K_actual), replace=False)], sub_cfg, device)

    model = MLPBioPD(in_dim=X_msd_sub.shape[1], hidden=sub_cfg.dense_units,
                    out_dim=sub_cfg.final_units, dropout=sub_cfg.dropout)
    if sub_cfg.try_compile and torch.cuda.is_available():
        try: model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except: pass

    # Train on the selected subset.
    print(f"\n[SUBSET TRAINING] Training on {K_actual} cells")
    model, sub_outputs = train_biopd_cell_small(
        model, X_msd_sub, psi_sub, sub_cfg, device, wins,
        full_sub_g, center_sub_g, cdeg,
        stage_codes_sub, dpt_sub_sorted)

    # Step 4: Predict ALL cells (chunked to avoid OOM)
    print(f"\n[FULL INFERENCE] Predicting {N} cells...")
    model.eval()
    X_full = X_msd_full.copy().astype(np.float32)
    batch = 4096
    parts = []
    with torch.no_grad():
        for i in range(0, N, batch):
            chunk = torch.as_tensor(X_full[i:i+batch], dtype=torch.float32, device=device)
            parts.append(model(chunk).cpu().numpy())
            del chunk
            torch.cuda.empty_cache()
        Y_all_sorted = np.concatenate(parts, axis=0)
    print(f"[FULL INFERENCE] Shape: {Y_all_sorted.shape}")

    # Step 5: Alignment with stage-annotation-dependent terms disabled.
    stage_codes_sorted_full = stage_codes[order].astype(np.int64)
    dpt_sorted_full = dpt_norm[order].astype(np.float32)

    try:
        adj_sorted_full = adata.obsp["connectivities"].tocsr()[order][:, order]
    except:
        adj_sorted_full = None

    outputs = [("dense3_full", Y_all_sorted)]

    if skip_alignment:
        print("[ALIGNMENT] Skipped (--skip_alignment)")
        outputs.append(("best", Y_all_sorted))
    else:
        # Alignment settings retain graph and pseudotime-derived regularization.
        align_kw = dict(n_epochs=150, lr=3e-4,
                        lambda_ord=0.0,         # Stage-annotation-dependent term disabled.
                        lambda_contr=0.0,       # Stage-annotation-dependent term disabled.
                        lambda_graph=0.3,       # Graph regularization.
                        lambda_flow=0.4,        # Pseudotime-flow regularization.
                        lambda_dpt_smooth=0.3,  # Pseudotime-neighborhood regularization.
                        lambda_compact=0.0,     # Stage-annotation-dependent term disabled.
                        n_edge_samples=30000)
        if override_lambda_graph is not None:
            align_kw["lambda_graph"] = override_lambda_graph
        if override_lambda_flow is not None:
            align_kw["lambda_flow"] = override_lambda_flow
        if override_lambda_dpt_smooth is not None:
            align_kw["lambda_dpt_smooth"] = override_lambda_dpt_smooth

        print(f"\n[ALIGNMENT] Alignment on {N} cells with stage-dependent losses disabled...")
        Y_best, sel, Y_d3, Y_al = global_alignment_finetune(
            model, X_full, stage_codes_sorted_full, dpt_sorted_full,
            adj_sorted_full, sub_cfg, device,
            batch_size=min(4096, N), **align_kw)

        outputs.append(("aligned", Y_al))
        outputs.append(("best", Y_d3 if sel.startswith("dense3") else Y_best))

    return model, outputs, order


# ======================================================================
#  MAIN
# ======================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda_device", type=str, default="0")
    p.add_argument("--tag", type=str, default="cell")
    p.add_argument("--out_dir", type=str, default="./results",
                   help="Base output directory")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--subset_k", type=int, default=8000,
                   help="Subset size for large data (N>=10K)")
    p.add_argument("--large_threshold", type=int, default=10000,
                   help="N >= this uses subset training")
    p.add_argument("--skip_alignment", action="store_true",
                   help="Skip alignment stage (use Dense3 directly)")
    p.add_argument("--lambda_graph", type=float, default=None,
                   help="Override graph smoothness lambda in alignment")
    p.add_argument("--lambda_flow", type=float, default=None,
                   help="Override pseudotime flow lambda in alignment")
    p.add_argument("--lambda_dpt_smooth", type=float, default=None,
                   help="Override DPT kNN smoothness lambda in alignment")
    args = p.parse_args()

    cfg = Cfg(); cfg.cuda_device = args.cuda_device; cfg.seed = args.seed; cfg.out_dir = args.out_dir
    setup_nature_style(); device = setup_env(cfg)

    OUT = os.path.join(cfg.out_dir, args.dataset, f"biopd_{args.tag}", f"seed{cfg.seed}")
    os.makedirs(OUT, exist_ok=True)
    print(f"\nBio-PD cell-level workflow: {args.dataset} → {OUT}")
    print(f"  Stage-annotation-dependent losses are disabled: contrastive, ordering, compactness, repulsion")
    print(f"  Active objectives include KL, graph regularization, pseudotime flow, and pseudotime-neighborhood smoothness")

    adata = sc.read_h5ad(args.h5ad)
    required_obs = ["stage", "dpt_pseudotime"]
    required_obsm = ["X_diffmap"]
    missing_obs = [k for k in required_obs if k not in adata.obs]
    missing_obsm = [k for k in required_obsm if k not in adata.obsm]
    if missing_obs or missing_obsm:
        raise ValueError(
            "Input .h5ad is missing required fields: "
            f"obs={missing_obs}, obsm={missing_obsm}. "
            "Required fields are adata.obs['stage'], adata.obs['dpt_pseudotime'], "
            "and adata.obsm['X_diffmap']."
        )
    adata.obs["stage"] = adata.obs["stage"].astype("category")
    t = np.asarray(adata.obs["dpt_pseudotime"]).astype(float)
    adata.obs["dpt_norm"] = (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t) + 1e-8)
    order = np.argsort(adata.obs["dpt_pseudotime"].values)
    N = adata.n_obs
    print(f"Loaded: {N} cells, threshold={args.large_threshold}")

    stage_names = list(adata.obs["stage"].cat.categories)
    eval_codes = adata.obs["stage"].cat.codes.to_numpy()
    dpt = adata.obs["dpt_norm"].to_numpy().astype(float)
    if "X_pca" not in adata.obsm: sc.tl.pca(adata, n_comps=50)
    X_pca = np.nan_to_num(np.asarray(adata.obsm["X_pca"]).copy())

    cfg.msd_dims = min(cfg.msd_dims, adata.obsm["X_diffmap"].shape[1])
    X_msd = build_msd_features(adata, order, n_dims=cfg.msd_dims, scales=cfg.msd_scales)
    psi_sorted = np.asarray(adata.obsm["X_diffmap"])[order, :cfg.msd_dims].astype(np.float32)
    np.nan_to_num(psi_sorted, copy=False)

    stage_codes_sorted = eval_codes[order].astype(np.int64)
    dpt_sorted = dpt[order].astype(np.float32)

    t0 = time.perf_counter()

    if N < args.large_threshold:
        # ---- SMALL DATA PATH ----
        print(f"\n[MODE] Small-data path (N={N} < {args.large_threshold}): direct training")
        adapt_cfg_to_dataset_size(cfg, N)
        make_cfg_label_free(cfg)  # Disable stage-annotation-dependent losses.

        B, C = int(cfg.batch_size), int(cfg.context)
        S = B if (cfg.center_nonoverlap or cfg.stride is None) else int(cfg.stride)
        wins = build_windows(N, B, C, S)
        full_sub, center_sub, cdeg = precompute_subgraphs_diffusion(psi_sorted, wins, C, cfg)
        if cfg.batch_size > 4096: cfg.P_backend = "cached_knn"; cfg.benchmark_on_start = False
        if cfg.benchmark_on_start and N > 100:
            cfg.P_backend = benchmark_P_backends(
                X_msd[np.random.choice(N, min(500, N), replace=False)], cfg, device)

        model = MLPBioPD(in_dim=X_msd.shape[1], hidden=cfg.dense_units,
                        out_dim=cfg.final_units, dropout=cfg.dropout)
        if cfg.try_compile and torch.cuda.is_available():
            try: model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except: pass

        model, outputs = train_biopd_cell_small(
            model, X_msd, psi_sorted, cfg, device, wins,
            full_sub, center_sub, cdeg,
            stage_codes_sorted, dpt_sorted)

        # Alignment with stage-annotation-dependent losses disabled.
        if args.skip_alignment:
            print("[ALIGNMENT] Skipped (--skip_alignment)")
            Yd3 = outputs[-1][1]  # Last stage output
            outputs.append(("best", Yd3))
        elif N < 200000:
            if N < 5000:
                akw = dict(n_epochs=100, lr=1e-3,
                           lambda_ord=0.0, lambda_contr=0.0, lambda_compact=0.0,
                           lambda_graph=0.5, lambda_flow=0.5, lambda_dpt_smooth=0.3,
                           n_edge_samples=10000)
            else:
                akw = dict(n_epochs=150, lr=5e-4,
                           lambda_ord=0.0, lambda_contr=0.0, lambda_compact=0.0,
                           lambda_graph=0.4, lambda_flow=0.5, lambda_dpt_smooth=0.3,
                           n_edge_samples=20000)
            # Apply optional command-line overrides.
            if args.lambda_graph is not None:
                akw["lambda_graph"] = args.lambda_graph
            if args.lambda_flow is not None:
                akw["lambda_flow"] = args.lambda_flow
            if args.lambda_dpt_smooth is not None:
                akw["lambda_dpt_smooth"] = args.lambda_dpt_smooth
            G_a = None
            try: G_a = adata.obsp["connectivities"].tocsr()[order][:, order]
            except: pass
            Yb, sel, Yd3, Yal = global_alignment_finetune(
                model, X_msd, stage_codes_sorted, dpt_sorted, G_a, cfg, device,
                batch_size=min(4096, N), **akw)
            outputs.append(("dense3_pre_align", Yd3))
            outputs.append(("aligned", Yal))
            outputs.append(("best", Yd3 if sel.startswith("dense3") else Yb))
    else:
        # ---- LARGE DATA PATH ----
        print(f"\n[MODE] Large-data path (N={N} >= {args.large_threshold}): subset K={args.subset_k}")
        model, outputs, order = train_biopd_cell_large(
            adata, X_msd, psi_sorted, order,
            eval_codes, dpt, cfg, device,
            subset_K=args.subset_k, seed=args.seed,
            skip_alignment=args.skip_alignment,
            override_lambda_graph=args.lambda_graph,
            override_lambda_flow=args.lambda_flow,
            override_lambda_dpt_smooth=args.lambda_dpt_smooth)

    elapsed = time.perf_counter() - t0

    # ---- SAVE & EVALUATE ----
    adata.uns["stage_colors"] = _fixed_stage_palette(adata.obs["stage"].cat.categories)
    biopd_stages = {}
    rows = []

    for name, Y in outputs:
        Yf = np.zeros_like(Y); Yf[order] = Y
        adata.obsm[f"X_biopd_{name}"] = Yf
        np.save(os.path.join(OUT, f"emb_biopd_{name}.npy"), Yf)
        biopd_stages[f"Bio-PD_{name}"] = Yf
        for cb in ["stage", "dpt_norm"]:
            fig = sc.pl.embedding(adata, basis=f"biopd_{name}", color=cb,
                                  legend_loc="right margin", frameon=False, size=6,
                                  show=False, return_fig=True)
            fig.savefig(os.path.join(OUT, f"embedding_{name}_{cb.replace('dpt_norm','dpt')}.png"),
                        dpi=300, bbox_inches="tight"); plt.close(fig)
        m = evaluate_embedding(Yf, X_pca, dpt, eval_codes, k=15, seed=0, max_cells=10000)
        m["Method"] = f"Bio-PD_{name}"; rows.append(m)

    df = pd.DataFrame(rows); df = df[["Method"] + [c for c in df.columns if c != "Method"]]
    df.to_csv(os.path.join(OUT, "per_stage_metrics.csv"), index=False)
    print("\n--- Bio-PD metrics ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save evaluation metadata for downstream analysis.
    emb_dir = os.path.join(OUT, "embeddings"); os.makedirs(emb_dir, exist_ok=True)
    np.save(os.path.join(emb_dir, "eval_stage_codes.npy"), eval_codes)
    np.save(os.path.join(emb_dir, "dpt_norm.npy"), dpt)

    Ydm = np.asarray(adata.obsm.get("X_diffmap", np.zeros((adata.n_obs, 4))))[:, 1:3].copy()
    all_b = {"DiffMap": Ydm}; all_b.update(biopd_stages)
    plot_stage_progression_biopd(all_b, eval_codes, dpt, stage_names, OUT)

    fn, fY = outputs[-1]; Yf = np.zeros_like(fY); Yf[order] = fY
    np.save(os.path.join(emb_dir, "emb_biopd_cell.npy"), Yf)
    adata.write(os.path.join(OUT, "biopd_cell_result.h5ad"))

    print(f"\nCompleted. {OUT} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
