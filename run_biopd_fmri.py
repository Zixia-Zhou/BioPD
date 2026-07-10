#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bio-PD fMRI workflow.

This standalone script implements the fMRI Bio-PD analysis pipeline, including
spatiotemporal preprocessing, optimal-transport spatial projection, Chebyshev
GCN refinement, multi-stage manifold optimization, embedding export, and KNN
scene-label evaluation.

Example:
  python run_biopd_fmri.py \
      --analysis full \
      --data_dir ./data \
      --labels_path ./data/sherlock_labels_coded_expanded.csv \
      --out_dir ./results/biopd_fmri \
      --gpu 0 \
      --seed 0
"""

import os
import sys

# Parse GPU selection before importing torch.
_GPU_ID = "0"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpu" and _i + 1 < len(sys.argv):
        _GPU_ID = sys.argv[_i + 1]
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_ID

import argparse
import math
import random
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
import sklearn
import sklearn.metrics as mpd
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numba
from numba import njit
import ot
from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import ChebConv

warnings.filterwarnings("ignore")


# ==============================================================================
# Sinkhorn optimal transport
# ==============================================================================

def sinkhorn(a, b, M, reg, method='sinkhorn', numItermax=1000,
             stopThr=1e-9, verbose=False, log=False, warn=False,
             **kwargs):
    
    return sinkhorn_knopp(a, b, M, reg, numItermax=numItermax,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warn=warn,
                              **kwargs)

def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True,
                   **kwargs):
   

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, n_hists), type_as=M) 
        v = nx.ones((dim_b, n_hists), type_as=M) 
    else:
        u = nx.ones(dim_a, type_as=M) 
        v = nx.ones(dim_b, type_as=M) 

    K = nx.exp(M / (-reg))
    u=u/max(u)
    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1. / nx.dot(Kp, v)

        if (nx.any(KtransposeU == 0)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum('ik,ij,jk->jk', u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum('i,ij,j->j', u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))
    else:
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    if log:
        log['niter'] = ii
        log['u'] = u
        log['v'] = v

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))


# ==============================================================================
# Gromov-Wasserstein optimization
# ==============================================================================

def tensor_square_loss_adjusted(C1, C2, T):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a**2) / 2

    def f2(b):
        return (b**2) / 2

    def h1(a):
        return a

    def h2(b):
        return b

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()
    return tens

def tensor_KL_loss_adjusted(C1, C2, T):

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a*np.log(a+1e-15)-a)

    def f2(b):
        return b

    def h1(a):
        return a

    def h2(b):
        return (np.log(b+1e-15))

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()

    return tens

def create_space_distributions(num_locations, num_cells):

    p_locations = ot.unif(num_locations)
    p_expression = ot.unif(num_cells)
    return p_locations, p_expression

def compute_random_coupling(p, q, epsilon):
    num_cells = len(p)
    num_locations = len(q)
    K = np.random.rand(num_cells, num_locations)
    C = -epsilon * np.log(K)
    return sinkhorn(p, q, C, epsilon,method='sinkhorn')

def gromov_wasserstein_adjusted_norm(cost_mat, C1, C2,p, q, loss_fun, epsilon,
                                     max_iter=1000, tol=1e-9, verbose=False, log=False, random_ini=False):
   
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)

    T = compute_random_coupling(p, q, epsilon) if random_ini else np.outer(p, q)  # Initialization

    cpt = 0
    err = 1
     
    while (err > tol and cpt < max_iter):
            
            Tprev = T

            if loss_fun == 'square_loss':
                tens = tensor_square_loss_adjusted(C1, C2, T)
            if loss_fun == 'kl_loss':
                    tens = tensor_KL_loss_adjusted(C1, C2, T)

            
            if epsilon ==0:
                T= ot.lp.emd(p, q, tens)
            else:
                T = sinkhorn(p, q, tens, epsilon,numItermax=max_iter)
        
            if cpt % 10 == 0:
            # We can speed up the process by checking for the error only all
            # the 10th iterations
                err = np.linalg.norm(T - Tprev)

                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}'.format(
                                'It.', 'Err') + '\n' + '-' * 19)
                        print('{:5d}|{:8e}|'.format(cpt, err))
            cpt += 1
    if log:
        return T, log
    else:
        return T


# ==============================================================================
# Spatial projection utilities
# ==============================================================================

def createMeshDistance(rowNum,colNum):
  
# If the row number is even
    if (rowNum % 2) == 0:
        Nx=rowNum/2
        x = np.linspace(-Nx, Nx-1, rowNum)
# If the row number is odd
    else:
        Nx=(rowNum-1)/2
        x = np.linspace(-Nx, Nx, rowNum)

# If the column number is even
    if (colNum % 2) == 0:
        Mx=colNum/2
        y = np.linspace(-Mx, Mx-1, colNum)
# If the column number is odd
    else:
       Mx=(colNum-1)/2
       y = np.linspace(-Mx, Mx, colNum)

# Create 2D mesh grid from 1D x and y grids
    xx, yy = np.meshgrid(x, y)
# Compute Euclidean distance between grid points
    zz = np.sqrt(xx**2 + yy**2)
# Make the 2D grid into a 1D vector and form the Euclidean distance matrix
    gridVec=zz.flatten()
    distMat=mpd.pairwise_distances(gridVec.reshape(-1,1))
    return distMat

def createInteractionMatrix(data, metric='correlation'):
    interactMat=mpd.pairwise_distances(data.T,metric=metric)
    return interactMat

def construct_neuromap(data,rowNum,colNum,epsilon=0,num_iter=1000):

    sizeData=data.shape
    numCell=sizeData[0]
    numGene=sizeData[1]
    # distance matrix of 2D genomap grid
    distMat = createMeshDistance(rowNum,colNum)
    # gene-gene interaction matrix 
    interactMat = createInteractionMatrix(data, metric='correlation')

    totalGridPoint=rowNum*colNum
    
    if (numGene<totalGridPoint):
        totalGridPointEff=numGene
    else:
        totalGridPointEff=totalGridPoint
    
    M = np.zeros((totalGridPointEff, totalGridPointEff))
    p, q = create_space_distributions(totalGridPointEff, totalGridPointEff)

   # Coupling matrix 
    T = gromov_wasserstein_adjusted_norm(
    M, interactMat, distMat[:totalGridPointEff,:totalGridPointEff], p, q, loss_fun='kl_loss', epsilon=epsilon,max_iter=num_iter)
 
    projMat = T*totalGridPoint
    # Data projected onto the couping matrix
    projM = np.matmul(data, projMat)

    neuromaps = np.zeros((numCell,rowNum, colNum, 1))

    px = np.asmatrix(projM)

    # Formation of neuromaps from the projected data
    for i in range(0, numCell-1):
        dx = px[i, :]
        fullVec = np.zeros((1,rowNum*colNum))
        fullVec[:dx.shape[0],:dx.shape[1]] = dx
        ex = np.reshape(fullVec, (rowNum, colNum), order='F').copy()
        neuromaps[i, :, :, 0] = ex
        
        
    return neuromaps


# ==============================================================================
# Data preprocessing utilities
# ==============================================================================

def TimeCORR(X, smooth_window=1):
    """Compute a time-correlation matrix M for X"""
    n_samples, n_features = X.shape
    A_feat = np.empty((n_samples, n_features))

    for f in range(n_features):
        A_feat[:, f] = sm.tsa.acf(X[:, f], fft=False, nlags=n_samples - 1, missing='drop')

    A_mean = np.nanmean(A_feat, axis=1)
    acf = np.convolve(A_mean, np.ones(smooth_window), 'same') / smooth_window

    drop_idx = np.where(acf < 0)[0]
    dropoff = n_samples if len(drop_idx) == 0 else drop_idx[0]

    M = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            lag = abs(i - j)
            if 0 < lag < dropoff:
                M[i, j] = acf[lag]
                M[j, i] = acf[lag]

    for row in M:
        s = np.sum(row)
        if s > 1e-12:
            row[:] /= s

    return M

def sherlock_para(data, balance_degree):
    n_all, _ = data.shape
    batch_size = n_all // balance_degree
    n = batch_size * balance_degree
    X_train = data[0:n, :]
    return X_train, batch_size, n

def spatiotemporal_projection(X_train, rowNum, colNum):
    autocorr_map = TimeCORR(X=X_train, smooth_window=1)
    X_train_auto = np.matmul(autocorr_map, X_train)

    nump = rowNum * colNum
    if nump < X_train_auto.shape[1]:
        selector = VarianceThreshold()
        selector.fit(X_train_auto)
        top_n_indices = selector.get_support(indices=True)
        X_train_auto = X_train_auto[:, top_n_indices[:nump]]

    NeuroMaps = construct_neuromap(X_train_auto, rowNum, colNum, epsilon=0.0, num_iter=200)
    return NeuroMaps


# ==============================================================================
# Manifold probability utilities
# ==============================================================================

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
        while np.abs(Hdiff) > tol and tries < 50:
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

@numba.jit(nopython=True)
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

def x2p_gpu(X, perplexity=30.0, tol=1e-5, max_iter=50):
    """
    GPU-accelerated perplexity-calibrated P matrix using PyTorch.
    ~5-10x faster than CPU version for n > 100.
    Falls back to CPU x2p if CUDA not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return x2p(X, tol=tol, perplexity=perplexity)
    except ImportError:
        return x2p(X, tol=tol, perplexity=perplexity)

    n = X.shape[0]
    device = torch.device('cuda')
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

    # Pairwise squared distances
    D_sq = torch.cdist(X_t, X_t, p=2).pow(2)
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    D_masked = D_sq[mask].view(n, n - 1)

    log_perp = np.log(perplexity)
    beta = torch.ones(n, device=device)

    for _ in range(max_iter):
        exp_D = torch.exp(-D_masked * beta.unsqueeze(1))
        sum_exp = exp_D.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P_row = exp_D / sum_exp
        H = -(P_row * (P_row + 1e-12).log()).sum(dim=1)
        H_diff = H - log_perp

        converged = H_diff.abs() < tol
        if converged.all():
            break
        beta = torch.where(H_diff > tol, beta * 1.5, beta)
        beta = torch.where(H_diff < -tol, beta / 1.5, beta)

    # Final P
    exp_D = torch.exp(-D_masked * beta.unsqueeze(1))
    sum_exp = exp_D.sum(dim=1, keepdim=True).clamp(min=1e-8)
    P_row = exp_D / sum_exp

    P_full = torch.zeros(n, n, device=device)
    P_full[mask] = P_row.flatten()

    return P_full.cpu().numpy()


# ==============================================================================
# Bio-PD neural model and evaluation utilities
# ==============================================================================

my_seed = 0

np.random.seed(my_seed)

random.seed(my_seed)

torch.manual_seed(my_seed)

torch.cuda.manual_seed(my_seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

def create_model(
        input_shape,
        num_conv_layers=4,
        filters_list=None,
        kernel_size=3,
        alpha=0.05,
        dense_units=(1024, 512, 256, 8),
        final_units=2
):
    """
    Create a CNN + Flatten + Dense architecture in PyTorch.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input, e.g. (rowNum, colNum).
    num_conv_layers : int
        Number of convolutional layers. Must be >= 1.
    filters_list : list of int, optional
        Number of filters for each conv layer. If None, defaults to [3, 16, 32, 64] for 4 conv layers.
    kernel_size : int or tuple
        Kernel size for all Conv2D layers.
    alpha : float
        Negative slope coefficient for LeakyReLU activations.
    dense_units : tuple of int
        Units for Dense layers. Must have at least three entries.
    final_units : int
        Number of output units, defaults to 2.

    Returns
    -------
    model : nn.Module
        The constructed PyTorch model.
    """

    # Provide default filters_list if not supplied
    if filters_list is None:
        filters_list = [3, 16, 32, 64]

    # Sanity checks
    if num_conv_layers < 1:
        raise ValueError("num_conv_layers must be >= 1.")
    if len(filters_list) != num_conv_layers:
        raise ValueError("filters_list length must match num_conv_layers.")
    if len(dense_units) < 3:
        raise ValueError("dense_units must have at least three elements for Dense1, Dense2, Dense3.")

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

            # Create convolutional layers
            self.conv_layers = nn.ModuleList()
            in_channels = 1  # Assuming grayscale input

            for i in range(num_conv_layers):
                conv = nn.Conv2d(in_channels, filters_list[i], kernel_size=kernel_size, padding='same')
                self.conv_layers.append(conv)
                in_channels = filters_list[i]

            self.leaky_relu = nn.LeakyReLU(alpha)
            self.flatten = nn.Flatten()

            # Calculate input size for first dense layer
            self.input_size = filters_list[-1] * input_shape[0] * input_shape[1]

            # Create dense layers
            self.dense_layers = nn.ModuleList()
            prev_units = self.input_size

            for units in dense_units:
                self.dense_layers.append(nn.Linear(prev_units, units))
                prev_units = units

            self.output_layer = nn.Linear(dense_units[-1], final_units)

        def forward(self, x):
            # Convolutional layers
            for conv in self.conv_layers:
                x = self.leaky_relu(conv(x))

            x = self.flatten(x)

            # Dense layers with stored intermediate outputs
            self.intermediate_outputs = []
            for i, dense in enumerate(self.dense_layers):
                x = F.relu(dense(x))
                if i < 3:  # Store first three dense layer outputs
                    self.intermediate_outputs.append(x)

            x = self.output_layer(x)
            return x

        def get_layer_output(self, x, layer_number):
            """
            Get output from specific dense layer (1-3).
            layer_number should be 1, 2, or 3 corresponding to Dense1, Dense2, Dense3
            """
            if not 1 <= layer_number <= 3:
                raise ValueError("layer_number must be 1, 2, or 3")

            # Forward through conv layers
            for conv in self.conv_layers:
                x = self.leaky_relu(conv(x))

            x = self.flatten(x)

            # Forward through dense layers until desired layer
            for i, dense in enumerate(self.dense_layers):
                x = F.relu(dense(x))
                if i + 1 == layer_number:
                    return x

            return x

    return MyModel()

def calculate_P(X, batch_size, HD_type):
    perplexity=30
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):

        if HD_type == 'sherlock':
            P_batch = x2p(X[i:i + batch_size], perplexity)
        elif HD_type == 'hippo':
            P_batch = x2p1(X[i:i + batch_size], perplexity)
        elif HD_type == 'monkey':
            P_batch = x2p2(X[i:i + batch_size], perplexity)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        P_batch = P_batch * 2  # Exaggerate
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12)
        P[i:i + batch_size] = P_batch

    return P

def calculate_P_optimized(X, batch_size, HD_type):
    """
    Optimized calculation of pairwise probabilities (P)
    """
    if HD_type != 'sherlock':
        raise ValueError(f"Unknown HD_type: {HD_type}")

    n = X.shape[0]
    P = np.zeros([n, n])

    # Process in batches
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        batch_data = X[i:end_idx]

        # Calculate distances
        D = prepare_distances(batch_data)

        # Calculate P for this batch
        P_batch = x2p_optimized(D, perplexity=30.0)

        # Clean and normalize P
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12)

        # Store in the full P matrix
        P[i:end_idx, i:end_idx] = P_batch

    return P

def evaluate_pos_monkey(pos_ref,pred):
    correlation_all=[]
    
    for i in range (0, 8):
        pos_ref1 =pos_ref[i*600:(i+1)*600]
        low_dim_data=pred[i*600:(i+1)*600]
        # pos_ref1 =pos_ref
        # low_dim_data=pred1
        ref_pos_distance_matrix = squareform(pdist(pos_ref1, metric='euclidean'))
    
        low_dim_distance_matrix = squareform(pdist(low_dim_data, metric='euclidean'))
        
        correlation, p_value = pearsonr(ref_pos_distance_matrix.flatten(), low_dim_distance_matrix.flatten())
    #    print(f"Position-Low's Pearson Correlation: {correlation}, P-value: {p_value}")
        correlation_all.append(correlation)
    correlation_all=np.array(correlation_all)

    ref_pos_distance_matrix = squareform(pdist(pos_ref, metric='euclidean'))
    
    low_dim_distance_matrix = squareform(pdist(pred, metric='euclidean'))
    
    correlation, p_value = pearsonr(ref_pos_distance_matrix.flatten(), low_dim_distance_matrix.flatten())
    
 #   print(f"Position-Low's Pearson Correlation: {correlation}, P-value: {p_value}")
    return correlation_all, correlation

def evaluate_pos_rat(pos_ref, pred):

    ref_pos_distance_matrix = squareform(pdist(pos_ref, metric='euclidean'))

    low_dim_distance_matrix = squareform(pdist(pred, metric='euclidean'))

    correlation, p_value = pearsonr(ref_pos_distance_matrix.flatten(), low_dim_distance_matrix.flatten())

    return correlation, p_value

def evaluate_pos_rat_spearman(pos_ref, pred):
    dist_pos = squareform(pdist(pos_ref, metric='euclidean'))
    dist_pred = squareform(pdist(pred, metric='euclidean'))

    rho, pval = spearmanr(dist_pos.flatten(), dist_pred.flatten())
    return rho, pval

def distance_covariance(data1, data2):
    # data1, data2 are shape (n, d1) and (n, d2).
    # Implement distance correlation steps:
    A = squareform(pdist(data1, 'euclidean'))
    B = squareform(pdist(data2, 'euclidean'))

    A_mean = A.mean(axis=0, keepdims=True)
    B_mean = B.mean(axis=0, keepdims=True)
    A_centered = A - A_mean - A_mean.T + A.mean()
    B_centered = B - B_mean - B_mean.T + B.mean()

    dcov = np.sqrt(np.mean(A_centered * B_centered))
    return dcov

def distance_correlation(data1, data2):
    dcovXY = distance_covariance(data1, data2)
    dcovXX = distance_covariance(data1, data1)
    dcovYY = distance_covariance(data2, data2)
    if dcovXX < 1e-12 or dcovYY < 1e-12:
        return 0
    return dcovXY / np.sqrt(dcovXX * dcovYY)

def evaluate_pos_rat_distance_corr(pos_ref, pred):
    # pos_ref shape (n, 1) or (n, 2) if 2D position
    # pred shape (n, d), embedding
    return distance_correlation(pos_ref, pred)

def mantel_test(matrix1, matrix2, permutations=1000):
    # matrix1, matrix2 are NxN distance matrices (no need to flatten upfront).
    # 1) compute real correlation
    real_r, _ = pearsonr(matrix1.flatten(), matrix2.flatten())

    # 2) permutations
    n = matrix1.shape[0]
    perm_r = []
    for _ in range(permutations):
        idx = np.random.permutation(n)
        # shuffle rows + columns of matrix2
        mat2_perm = matrix2[idx][:, idx]
        r_perm, _ = pearsonr(matrix1.flatten(), mat2_perm.flatten())
        perm_r.append(r_perm)

    perm_r = np.array(perm_r)
    # p-value: fraction of permutations that exceed real correlation
    p_value = np.mean(perm_r >= real_r)

    return real_r, p_value, perm_r  # or just real_r, p_value

def evaluate_pos_rat_mantel(pos_ref, pred, permutations=1000):
    dist_pos = squareform(pdist(pos_ref, metric='euclidean'))
    dist_pred = squareform(pdist(pred, metric='euclidean'))
    real_r, p_value, distribution = mantel_test(dist_pos, dist_pred, permutations)
    return real_r, p_value


# ==============================================================================
# Bio-PD fMRI workflow
# ==============================================================================

class Config:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
        else:
            self.set_defaults()

    def load_from_yaml(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        env = cfg.get('environment', {})
        self.cuda_device = env.get('cuda_device', '3')
        self.seed = env.get('seed', 0)

        train = cfg.get('training', {})
        self.n_iterations = train.get('n_iterations', 2)
        self.n_recur = train.get('n_recur', 4)
        self.balance_degree = train.get('balance_degree', 4)
        self.batch_size = train.get('batch_size', 4)
        self.max_epochs = train.get('max_epochs', 150)
        self.patience = train.get('patience', 30)
        self.learning_rate = train.get('learning_rate', 1e-3)
        self.max_grad_norm = train.get('max_grad_norm', 5.0)
        self.weight_decay = train.get('weight_decay', 1e-5)

        # Overlap-temporal params
        self.lambda_temporal = train.get('lambda_temporal', 0.2)
        self.temporal_max_lag = train.get('temporal_max_lag', 5)
        self.smooth_window = train.get('smooth_window', 1)
        self.context = train.get('context', 64)
        self.stride = train.get('stride', None)  # if None, will set to batch_size // 2

        # Optional epoch-offset schedule
        shift = cfg.get('shift_windows', {})
        self.shift_enabled = shift.get('enabled', True)
        self.shift_step = shift.get('step', 50)
        self.shift_num_offsets = shift.get('num_offsets', 5)
        self.p_update_frequency = shift.get('p_update_frequency', 20)

        model = cfg.get('model_architecture', {})
        self.num_conv_layers = model.get('num_conv_layers', 4)
        self.filters_list = model.get('filters_list', [3, 16, 32, 64])
        self.kernel_size = model.get('kernel_size', 3)
        self.alpha = model.get('alpha', 0.05)
        self.dense_units = tuple(model.get('dense_units', [1024, 512, 256, 8]))
        self.final_units = model.get('final_units', 2)

        cheb = cfg.get('cheb_gcn_refinement', {})
        self.cheb_K = cheb.get('K', 3)
        self.cheb_hidden_dim = cheb.get('hidden_dim', 32)
        self.cheb_out_channels = cheb.get('out_channels', 8)
        self.cheb_lr = cheb.get('learning_rate', 1e-3)
        self.cheb_epochs = cheb.get('epochs', 50)
        self.use_timecorr = cheb.get('use_timecorr', True)
        self.top_k_correlations = cheb.get('top_k_correlations', 8)

        data = cfg.get('data', {})
        self.HD_type = data.get('HD_type', 'sherlock')
        self.low_dim = data.get('low_dim', 2)
        self.perplexity = data.get('perplexity', 30)

        roi_list = cfg.get('roi_info', [])
        self.roi_info = {}
        for idx, roi in enumerate(roi_list):
            self.roi_info[idx] = roi

        paths = cfg.get('paths', {})
        self.base_data_path = paths.get('base_data_path', './data')
        self.base_model_save_dir = paths.get('base_model_save_dir', './results/biopd_fmri')
        self.base_plot_save_dir = paths.get('base_plot_save_dir', './results/biopd_fmri')
        self.labels_path = paths.get('labels_path', './data/sherlock_labels_coded_expanded.csv')

        subj = cfg.get('subject_range', {})
        self.subject_start = subj.get('start', 1)
        self.subject_end = subj.get('end', 17)

        viz = cfg.get('visualization', {})
        self.figure_size = tuple(viz.get('figure_size', [15, 10]))
        self.font_size = viz.get('font_size', 28)
        self.marker_size = viz.get('marker_size', 18)

        eval_params = cfg.get('evaluation', {})
        self.knn_k_values = eval_params.get('knn_k_values', [1, 3, 5, 8, 10, 30])
        self.cv_folds = eval_params.get('cross_validation_folds', 10)

    def set_defaults(self):
        self.cuda_device = "3"
        self.seed = 0
        self.n_iterations = 2
        self.n_recur = 4
        self.balance_degree = 4
        self.batch_size = 4
        self.max_epochs = 150
        self.patience = 30
        self.learning_rate = 1e-3
        self.max_grad_norm = 5.0
        self.weight_decay = 1e-5

        # Overlap-temporal defaults
        self.lambda_temporal = 0.2
        self.temporal_max_lag = 5
        self.smooth_window = 1
        self.temporal_loss_type = 'velocity'  # 'velocity' (L2 on 1st deriv) or 'curvature' (L2 on 2nd deriv)
        self.post_smooth_sigma = 3  # Gaussian temporal smoothing on final embedding (0 = disabled)
        self.context = 64
        self.stride = None  # set later to batch_size // 2

        # Boundary-aware temporal loss
        self.use_event_boundary_loss = False
        self.lambda_event = 0.3
        self.event_max_lag = 20
        self.event_boundary_percentile = 75
        self.event_boundary_min_stage = 1  # 1=all stages, 3=only stages 3-4, etc.

        # Temporal distance correlation loss (preserve high-D / low-D distance patterns)
        self.use_tdist_loss = False
        self.lambda_tdist = 0.1
        self.use_ratio_loss = False
        self.lambda_ratio = 0.1
        self.tdist_warmup_epochs = 10  # skip first N epochs for gradient stability

        # Directional consistency loss (anti-zigzag)
        self.use_directional_loss = False
        self.lambda_directional = 0.1

        # Offset schedule
        self.shift_enabled = True
        self.shift_step = 50
        self.shift_num_offsets = 5
        self.p_update_frequency = 20

        # Model
        self.num_conv_layers = 4
        self.filters_list = [3, 16, 32, 64]
        self.kernel_size = 3
        self.alpha = 0.05
        self.dense_units = (1024, 512, 256, 8)
        self.final_units = 2

        # Cheb refine
        self.cheb_K = 3
        self.cheb_hidden_dim = 32
        self.cheb_out_channels = 8
        self.cheb_lr = 1e-3
        self.cheb_epochs = 50
        self.cheb_temporal_mode = 'spatial_only'  # 'spatial_only', 'temporal_features', 'sliding_window'
        self.cheb_temporal_radius = 2  # temporal window radius for temporal_features / sliding_window
        self.use_timecorr = True
        self.top_k_correlations = 8

        # Data
        self.HD_type = 'sherlock'
        self.low_dim = 2
        self.perplexity = 30

        # ROI (same as before)
        self.roi_info = {
            0: {"name": "HV", "model_dir": "HV_models", "data_file": "high_Visual_sherlock_movie.npy"},
            1: {"name": "EA", "model_dir": "EA_models", "data_file": "aud_early_sherlock_movie.npy"},
            2: {"name": "EV", "model_dir": "EV_models", "data_file": "early_visual_sherlock_movie.npy"},
            3: {"name": "PMC", "model_dir": "PMC_models", "data_file": "pmc_nn_sherlock_movie.npy"}
        }

        # Paths
        self.base_data_path = "./data"
        self.base_model_save_dir = "./results/biopd_fmri"
        self.base_plot_save_dir = "./results/biopd_fmri"
        self.labels_path = './data/sherlock_labels_coded_expanded.csv'

        # Subjects
        self.subject_start = 1
        self.subject_end = 17

        # Viz & Eval
        self.figure_size = (15, 10)
        self.font_size = 28
        self.marker_size = 18
        self.knn_k_values = [1, 3, 5, 8, 10, 30]
        self.cv_folds = 10

        # Analysis-mode settings
        self.use_gw_layout = False
        self.use_roi_adaptive = False
        self.run_name = "standard"

        # ROI-adaptive parameter overrides
        self.roi_adaptive_params = {
            "HV": {"lambda_temporal": 0.20, "perplexity": 30, "context": 64,
                    "patience": 25, "cheb_K": 3, "top_k_correlations": 8},
            "EA": {"lambda_temporal": 0.15, "perplexity": 28, "context": 64,
                    "patience": 30, "cheb_K": 3, "top_k_correlations": 8},
            "EV": {"lambda_temporal": 0.28, "perplexity": 18, "context": 48,
                    "patience": 30, "cheb_K": 2, "top_k_correlations": 8},
            "PMC": {"lambda_temporal": 0.10, "perplexity": 22, "context": 40,
                     "patience": 38, "cheb_K": 4, "top_k_correlations": 10},
        }

def compute_gw_projection(data_2d, H, W):
    """Compute GW optimal transport projection matrix from data.
    Returns projMat of shape (S, min(S, H*W)) and order='F' reshape info."""
    T, S = data_2d.shape
    totalGridPoint = H * W
    totalGridPointEff = min(S, totalGridPoint)

    distMat = createMeshDistance(H, W)
    interactMat = createInteractionMatrix(data_2d, metric='correlation')

    p, q = create_space_distributions(totalGridPointEff, totalGridPointEff)
    M_init = np.zeros((totalGridPointEff, totalGridPointEff))

    T_coupling = gromov_wasserstein_adjusted_norm(
        M_init,
        interactMat[:totalGridPointEff, :totalGridPointEff],
        distMat[:totalGridPointEff, :totalGridPointEff],
        p, q, loss_fun='kl_loss', epsilon=0.0, max_iter=200
    )
    projMat = T_coupling * totalGridPoint
    return projMat

def apply_gw_projection(data_2d, projMat, H, W):
    """Project data using precomputed GW coupling and reshape to 3D."""
    T, S = data_2d.shape
    projM = data_2d @ projMat  # (T, totalGridPointEff)
    data_3d = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        fullVec = np.zeros(H * W)
        L = min(projM.shape[1], H * W)
        fullVec[:L] = projM[t, :L]
        data_3d[t] = np.reshape(fullVec, (H, W), order='F')
    return data_3d

def setup_environment(config):
    # CUDA_VISIBLE_DEVICES already set at top of file
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Enable full determinism for reproducibility (warn_only=True to avoid crashes)
    if getattr(config, 'use_deterministic', False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
    return device

def get_epoch_offset(epoch, config):
    if not config.shift_enabled or config.batch_size <= 1:
        return 0
    # Decouple offset schedule from P-cache frequency:
    # Offset changes every shift_step epochs (default: every epoch for max diversity)
    shift_freq = getattr(config, 'shift_frequency', 1)
    idx = (epoch // max(1, shift_freq)) % max(1, config.shift_num_offsets)
    step = min(config.shift_step, max(1, config.batch_size - 1))
    return int((idx * step) % config.batch_size)

def calculate_P_improved(X, batch_size, HD_type, perplexity=30, offset=0, include_tail=True):
    """
    Shifted-batch P for a contiguous block X (n_block x d).
    With batch_size == n_block, this returns a single P.
    """
    n = X.shape[0]
    P_list = []
    if n < 2:
        return P_list
    start = int(offset % max(1, batch_size))
    i = start
    while i < n:
        end = min(i + batch_size, n)
        if end - i < 2:
            break
        block = X[i:end]
        if HD_type == 'sherlock':
            P_batch = x2p(block, perplexity)
        elif HD_type == 'hippo':
            P_batch = x2p1(block, perplexity)
        elif HD_type == 'monkey':
            P_batch = x2p2(block, perplexity)
        else:
            raise ValueError(f"Unknown HD_type: {HD_type}")
        P_batch[np.isnan(P_batch)] = 0
        P_batch = (P_batch + P_batch.T) / 2.0
        P_batch = P_batch / (P_batch.sum() + 1e-8)
        P_batch = np.maximum(P_batch, 1e-12)
        P_list.append(P_batch)
        i += batch_size
        if not include_tail and i >= n:
            break
    return P_list

def create_kl_divergence_stable(low_dim=2):
    def KLdivergence(P, Y):
        alpha = low_dim - 1
        eps = 1e-15
        sum_Y = torch.sum(Y ** 2, dim=1)
        D = sum_Y[:, None] + sum_Y[None, :] - 2 * (Y @ Y.t())
        D = torch.clamp(D, min=0)
        Q = torch.pow(1 + D / alpha, -(alpha + 1) / 2)
        mask = 1 - torch.eye(Y.shape[0], device=Y.device)
        Q = Q * mask
        Q_sum = torch.sum(Q)
        if Q_sum > eps:
            Q = Q / Q_sum
        else:
            Q = mask / (Y.shape[0] * (Y.shape[0] - 1))
        Q = torch.clamp(Q, min=eps, max=1.0)
        mask_p = P > eps
        kl = torch.zeros_like(P)
        kl[mask_p] = P[mask_p] * torch.log(P[mask_p] / (Q[mask_p] + eps))
        return torch.sum(kl)
    return KLdivergence

def make_acf_weights(X, max_lag=5, smooth_window=1):
    """
    X: (n, d) real-valued series, compute mean ACF over features for first max_lag lags.
    Returns torch tensor of shape (max_lag,)
    """
    n, f = X.shape
    acfs = []
    for k in range(f):
        acf_k = sm.tsa.acf(X[:, k], fft=False, nlags=max_lag, missing='drop')
        acfs.append(acf_k)
    acf_mean = np.nanmean(np.stack(acfs, axis=0), axis=0)  # (max_lag+1,)
    w = np.clip(acf_mean[1:], 0, None)  # drop lag 0 and clamp negatives
    if smooth_window > 1:
        w = np.convolve(w, np.ones(smooth_window), 'same') / smooth_window
    w = w / (w.sum() + 1e-8)
    return torch.tensor(w, dtype=torch.float32)

def post_smooth_embedding(emb, sigma):
    """Apply Gaussian temporal smoothing to final embedding to remove zigzag."""
    if sigma <= 0:
        return emb
    return np.column_stack([gaussian_filter1d(emb[:, d], sigma=sigma)
                            for d in range(emb.shape[1])])

def temporal_smoothness_loss(Y, acf_w):
    """
    Y: (B_full, d) embeddings for the full window
    acf_w: (L,) weights for lags 1..L
    Computes: sum_{lag=1..L} w_lag * mean((Y[:-lag] - Y[lag:])^2)
    """
    B = Y.shape[0]
    L = int(acf_w.numel())
    if L == 0 or B < 2:
        return Y.new_tensor(0.0)
    loss = Y.new_tensor(0.0)
    for lag in range(1, L + 1):
        if B - lag <= 0:
            break
        diff = Y[:-lag] - Y[lag:]
        loss = loss + acf_w[lag - 1] * (diff.pow(2).mean())
    return loss

def temporal_smoothness_curvature(Y, acf_w):
    """
    Penalise the *second derivative* (curvature / acceleration) of the
    embedding trajectory instead of the first derivative.

    Zigzag artefacts have very high curvature (rapid direction reversals)
    while genuine event transitions are smooth single direction changes.

    Y:     (B, d) embeddings for the full window
    acf_w: (L,) weights for lags 1..L  (re-used from ACF for consistency)

    For each lag k, the discrete second derivative is:
        a_t = Y[t+k] - 2*Y[t] + Y[t-k]
    We minimise  sum_k  w_k * mean(|a_t|^2)
    """
    B = Y.shape[0]
    L = int(acf_w.numel())
    if L == 0 or B < 3:
        return Y.new_tensor(0.0)
    loss = Y.new_tensor(0.0)
    for lag in range(1, L + 1):
        if B - 2 * lag <= 0:
            break
        # second derivative: Y[t+lag] - 2*Y[t] + Y[t-lag]
        accel = Y[2*lag:] - 2 * Y[lag:-lag] + Y[:-(2*lag)]
        loss = loss + acf_w[lag - 1] * accel.pow(2).mean()
    return loss

def temporal_smoothness_huber(Y, acf_w, delta=1.0):
    """
    Huber loss on 1st derivative: L2 for small jitter (|diff| < delta),
    L1 for large jumps (event transitions).
    Smooths zigzag without over-penalising genuine transitions.
    """
    B = Y.shape[0]
    L = int(acf_w.numel())
    if L == 0 or B < 2:
        return Y.new_tensor(0.0)
    loss = Y.new_tensor(0.0)
    for lag in range(1, L + 1):
        if B - lag <= 0:
            break
        diff = Y[:-lag] - Y[lag:]
        # Per-sample norm for Huber threshold
        diff_norm = diff.pow(2).sum(dim=-1, keepdim=True).sqrt()
        # Huber: L2 where norm < delta, L1 where norm >= delta
        is_small = (diff_norm < delta).float()
        huber = is_small * diff.pow(2) + (1 - is_small) * (delta * diff.abs() - 0.5 * delta**2)
        loss = loss + acf_w[lag - 1] * huber.mean()
    return loss

def _temporal_loss(Y, acf_w, config):
    """Dispatch to velocity or curvature temporal loss based on config."""
    mode = getattr(config, 'temporal_loss_type', 'velocity')
    if mode == 'curvature':
        return temporal_smoothness_curvature(Y, acf_w)
    elif mode == 'hybrid':
        # Combine velocity (stability) + curvature (anti-zigzag)
        vel = temporal_smoothness_loss(Y, acf_w)
        curv = temporal_smoothness_curvature(Y, acf_w)
        alpha = getattr(config, 'hybrid_curvature_weight', 0.5)
        return (1 - alpha) * vel + alpha * curv
    elif mode == 'huber':
        delta = getattr(config, 'huber_delta', 1.0)
        return temporal_smoothness_huber(Y, acf_w, delta=delta)
    return temporal_smoothness_loss(Y, acf_w)

def compute_boundary_weights(X_flat, smooth_window=3, percentile=75):
    """
    Detect event boundaries from high-D temporal gradients (unsupervised).
    Returns weights in [0, 1]: high = within-event (smooth), low = boundary (allow jump).

    X_flat: (n, d) high-dimensional data
    Returns: numpy (n-1,) float32
    """
    # Consecutive-timepoint Euclidean distances in high-D
    gradients = np.sqrt(np.sum((X_flat[1:] - X_flat[:-1]) ** 2, axis=1))

    # Smooth to reduce noise
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        gradients = uniform_filter1d(gradients, smooth_window)

    # Sigmoid: large gradient → low weight (boundary), small gradient → high weight
    threshold = np.percentile(gradients, percentile)
    scale = np.std(gradients) + 1e-8
    weights = 1.0 / (1.0 + np.exp((gradients - threshold) / (0.3 * scale)))

    return weights.astype(np.float32)

def boundary_aware_temporal_loss(Y, bw_slice, max_lag=20):
    """
    Temporal smoothness weighted by boundary detection.
    Uses rolling-min of boundary weights so that ANY boundary between t and t+lag
    reduces the smoothing penalty.

    Y: (B, d) embeddings for a window
    bw_slice: (B-1,) boundary weights for this window (torch tensor on device)
    max_lag: max temporal lag
    """
    B = Y.shape[0]
    if B < 2 or bw_slice.numel() == 0:
        return Y.new_tensor(0.0)

    max_lag = min(max_lag, B - 1)
    loss = Y.new_tensor(0.0)
    total_weight = 0.0

    # Precompute rolling-min using max_pool1d on negated values
    neg_bw = (-bw_slice).unsqueeze(0).unsqueeze(0)  # (1, 1, B-1)

    for lag in range(1, max_lag + 1):
        if B - lag <= 0:
            break
        diff = Y[:-lag] - Y[lag:]  # (B-lag, d)

        # Rolling min of boundary weights over window of size `lag`
        if lag == 1:
            w = bw_slice[:B - 1]
        else:
            # max_pool1d on negated = rolling min
            neg_min = F.max_pool1d(neg_bw, kernel_size=lag, stride=1)  # (1, 1, B-lag)
            w = -neg_min.squeeze(0).squeeze(0)  # (B-lag,)

        # Ensure shapes match
        n_pairs = diff.shape[0]
        w = w[:n_pairs]

        lag_decay = 1.0 / lag  # Natural decay with distance
        weighted_loss = (w.unsqueeze(1) * diff.pow(2)).mean()
        loss = loss + lag_decay * weighted_loss
        total_weight += lag_decay

    return loss / (total_weight + 1e-8)

def boundary_contrast_loss(Y, bw_slice, margin=1.0):
    """
    Contrastive loss at event boundaries: push apart embeddings across boundaries,
    pull together embeddings within events.

    bw_slice: (B-1,) boundary weights — LOW values = boundary, HIGH values = within-event
    """
    B = Y.shape[0]
    if B < 2 or bw_slice.numel() == 0:
        return Y.new_tensor(0.0)

    n_pairs = min(B - 1, bw_slice.shape[0])
    diff_sq = (Y[:n_pairs] - Y[1:n_pairs + 1]).pow(2).sum(dim=1)  # (n_pairs,)

    w = bw_slice[:n_pairs]
    # boundary_strength: high at boundaries (where bw is low)
    boundary_strength = 1.0 - w

    # Within-event (high w): minimize distance → w * dist^2
    L_within = (w * diff_sq).mean()

    # Across-boundary (high boundary_strength): maximize distance → boundary_strength * max(0, margin - dist)^2
    dist = diff_sq.sqrt()
    L_across = (boundary_strength * F.relu(margin - dist).pow(2)).mean()

    return L_within + L_across

def directional_consistency_loss(Y):
    """
    Penalize direction changes between consecutive velocity vectors.
    Scale-invariant: only penalizes direction reversals, not speed changes.
    This directly targets zigzag artifacts without suppressing event transitions.

    Y: (B, d) embeddings
    Returns: scalar loss (lower = smoother direction)
    """
    vel = Y[1:] - Y[:-1]  # (B-1, d)
    if vel.shape[0] < 2:
        return Y.new_tensor(0.0)
    # Cosine similarity between consecutive velocity vectors
    cos_sim = F.cosine_similarity(vel[:-1], vel[1:], dim=1)  # (B-2,)
    # Minimize negative cosine similarity = maximize directional consistency
    return -cos_sim.mean()

def temporal_distance_correlation_loss(Y, d_high_slice):
    """
    Maximize Pearson correlation between high-D and low-D consecutive distances.
    This directly optimizes the temporal_corr metric.

    Y: (B, d) low-D embeddings for a window
    d_high_slice: (B-1,) precomputed high-D consecutive Euclidean distances
    Returns: negative Pearson correlation (minimize to maximize correlation)
    """
    d_low = torch.sqrt(torch.sum((Y[1:] - Y[:-1]) ** 2, dim=1) + 1e-8)
    n = d_low.shape[0]
    if n < 3:
        return Y.new_tensor(0.0)
    d_high = d_high_slice[:n]
    mu_h = d_high.mean()
    mu_l = d_low.mean()
    dh = d_high - mu_h
    dl = d_low - mu_l
    std_h = torch.sqrt((dh ** 2).mean() + 1e-8)
    std_l = torch.sqrt((dl ** 2).mean() + 1e-8)
    corr = (dh * dl).mean() / (std_h * std_l)
    return -corr  # minimize negative corr = maximize correlation

def distance_ratio_preservation_loss(Y, d_high_slice):
    """
    Preserve relative magnitudes of consecutive temporal transitions.
    Normalizes both high-D and low-D distances by their means, then MSE.

    Y: (B, d) low-D embeddings
    d_high_slice: (B-1,) precomputed high-D consecutive distances
    Returns: MSE between normalized distance profiles
    """
    d_low = torch.sqrt(torch.sum((Y[1:] - Y[:-1]) ** 2, dim=1) + 1e-8)
    n = d_low.shape[0]
    if n < 3:
        return Y.new_tensor(0.0)
    d_high = d_high_slice[:n]
    ratio_high = d_high / (d_high.mean() + 1e-8)
    ratio_low = d_low / (d_low.mean() + 1e-8)
    return F.mse_loss(ratio_low, ratio_high)

class ChebRefineModel(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, out_channels=16, K=3):
        super().__init__()
        self.cheb1 = ChebConv(in_channels, hidden_dim, K=K)
        self.cheb2 = ChebConv(hidden_dim, out_channels, K=K)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.cheb1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.cheb2(x, edge_index, edge_weight)
        return x

def build_edges_vectorized(T, H, W, top_k_correlations, data_hw):
    """Build a full spatiotemporal graph for all time points."""
    edges = []
    t_coords, i_coords, j_coords = np.meshgrid(
        np.arange(T), np.arange(H), np.arange(W), indexing='ij'
    )
    node_ids = t_coords * (H * W) + i_coords * W + j_coords
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni = i_coords + di
        nj = j_coords + dj
        valid = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
        src = node_ids[valid].ravel()
        dst = (t_coords[valid] * (H * W) + ni[valid] * W + nj[valid]).ravel()
        edges.extend(zip(src, dst))
        edges.extend(zip(dst, src))
    for t in range(T - 1):
        cur = np.arange(t * H * W, (t + 1) * H * W)
        nxt = cur + H * W
        edges.extend(zip(cur, nxt))
        edges.extend(zip(nxt, cur))
    P = H * W
    corr = np.corrcoef(data_hw, rowvar=False)
    np.fill_diagonal(corr, -999)
    abs_corr = np.abs(corr)
    if top_k_correlations < P:
        top_idx = np.argpartition(-abs_corr, top_k_correlations, axis=1)[:, :top_k_correlations]
    else:
        top_idx = np.tile(np.arange(P), (P, 1))
    p_arr = np.repeat(np.arange(P), top_k_correlations)
    q_arr = top_idx.ravel()
    mask = p_arr != q_arr
    p_arr, q_arr = p_arr[mask], q_arr[mask]
    n1 = (np.arange(T)[:, None] * P + p_arr[None, :]).ravel()
    n2 = (np.arange(T)[:, None] * P + q_arr[None, :]).ravel()
    edges.extend(zip(n1, n2))
    edges.extend(zip(n2, n1))
    return np.asarray(edges, dtype=np.int64).T

def build_spatial_edges(H, W, top_k_correlations, data_hw):
    """
    Build H*W spatial graph (shared across all timepoints).
    Grid adjacency + correlation-based edges. No temporal edges.
    Returns edge_index (2, E) for a single H*W spatial graph.
    """
    P = H * W
    edges = []
    # Grid adjacency (4-connected)
    for i in range(H):
        for j in range(W):
            node = i * W + j
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    edges.append((node, ni * W + nj))
    # Correlation-based edges
    corr = np.corrcoef(data_hw, rowvar=False)
    np.fill_diagonal(corr, -999)
    abs_corr = np.abs(corr)
    if top_k_correlations < P:
        top_idx = np.argpartition(-abs_corr, top_k_correlations, axis=1)[:, :top_k_correlations]
    else:
        top_idx = np.tile(np.arange(P), (P, 1))
    for p in range(P):
        for q in top_idx[p]:
            if p != q:
                edges.append((p, q))
                edges.append((q, p))
    # Deduplicate
    edges = list(set(edges))
    return np.asarray(edges, dtype=np.int64).T

def build_sliding_window_edges(spatial_edges, P, window_size):
    """
    Build edge index for a sliding window graph: window_size copies of the
    spatial graph stacked with temporal edges between adjacent time-slices.
    Returns edge_index (2, E) for one window of window_size * P nodes.
    Node layout: time_offset * P + spatial_node.
    """
    edges = []
    # Spatial edges within each time-slice
    for t in range(window_size):
        offset = t * P
        for i in range(spatial_edges.shape[1]):
            edges.append((spatial_edges[0, i] + offset, spatial_edges[1, i] + offset))
    # Temporal edges: connect same spatial node across adjacent time-slices
    for t in range(window_size - 1):
        for p in range(P):
            src = t * P + p
            dst = (t + 1) * P + p
            edges.append((src, dst))
            edges.append((dst, src))
    return np.asarray(edges, dtype=np.int64).T

def refine_data_with_ChebGCN(data_2d, config, device, gw_proj_mat=None, cheb_warmstart=None):
    if config.use_timecorr:
        # simple time corr smoothing
        n_samples, _ = data_2d.shape
        A_feat = np.empty((n_samples, data_2d.shape[1]))
        for f in range(data_2d.shape[1]):
            A_feat[:, f] = sm.tsa.acf(data_2d[:, f], fft=False, nlags=n_samples - 1, missing='drop')
        A_mean = np.nanmean(A_feat, axis=1)
        acf = np.convolve(A_mean, np.ones(config.smooth_window), 'same') / config.smooth_window
        drop_idx = np.where(acf < 0)[0]
        dropoff = n_samples if len(drop_idx) == 0 else drop_idx[0]
        # Vectorized M_corr construction (replaces O(n²) Python loop)
        row_idx = np.arange(n_samples)
        lag_matrix = np.abs(row_idx[:, None] - row_idx[None, :])
        M = np.zeros((n_samples, n_samples), dtype=np.float32)
        valid = (lag_matrix > 0) & (lag_matrix < dropoff)
        M[valid] = acf[lag_matrix[valid]]
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        M /= row_sums
        data_2d = M @ data_2d

    T, S = data_2d.shape
    H = int(math.floor(math.sqrt(S)))
    W = int(math.ceil(S / H))

    if gw_proj_mat is not None:
        data_3d = apply_gw_projection(data_2d, gw_proj_mat, H, W)
        print(f"  [GW] Applied GW spatial layout: ({T}, {S}) -> ({T}, {H}, {W})")
    else:
        data_3d = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            row = data_2d[t]
            L = min(H * W, S)
            data_3d[t].flat[:L] = row[:L]

    P = H * W
    data_hw = data_3d.reshape(T, -1)

    # Build spatial edges (shared by all modes)
    spatial_edges_np = build_spatial_edges(H, W, config.top_k_correlations, data_hw)

    temporal_mode = getattr(config, 'cheb_temporal_mode', 'spatial_only')
    temporal_radius = getattr(config, 'cheb_temporal_radius', 2)
    window_size = 2 * temporal_radius + 1

    # ---- Determine input features and graph structure per mode ----
    if temporal_mode == 'temporal_features':
        # Mode 1: Spatial graph + temporal context as node features
        # Each node gets features [x(t-k), ..., x(t), ..., x(t+k)]
        n_feat = window_size
        x_np = np.zeros((T, P, n_feat), dtype=np.float32)
        data_flat = data_3d.reshape(T, P)
        for dt in range(-temporal_radius, temporal_radius + 1):
            src_idx = np.clip(np.arange(T) + dt, 0, T - 1)
            x_np[:, :, dt + temporal_radius] = data_flat[src_idx]
        # Target: reconstruct center timepoint only
        x_target_np = data_flat[:, :, np.newaxis].astype(np.float32)  # (T, P, 1)
        edge_index_t = torch.as_tensor(spatial_edges_np, dtype=torch.long, device=device)
        graph_P = P  # nodes per graph instance
        print(f"  [ChebGCN] temporal_features mode: {P} nodes, {spatial_edges_np.shape[1]} edges, "
              f"{n_feat} features/node (radius={temporal_radius})")

    elif temporal_mode == 'sliding_window':
        # Mode 2: Local spatio-temporal graph per window
        n_feat = 1
        data_flat = data_3d.reshape(T, P)
        # Build window edge index (window_size * P nodes)
        window_edges_np = build_sliding_window_edges(spatial_edges_np, P, window_size)
        edge_index_t = torch.as_tensor(window_edges_np, dtype=torch.long, device=device)
        graph_P = window_size * P  # nodes per graph instance
        # Prepare windowed data: for each center t, gather t-k...t+k
        x_np = np.zeros((T, window_size, P, 1), dtype=np.float32)
        x_target_np = data_flat[:, :, np.newaxis].astype(np.float32)  # (T, P, 1)
        for dt in range(-temporal_radius, temporal_radius + 1):
            src_idx = np.clip(np.arange(T) + dt, 0, T - 1)
            x_np[:, dt + temporal_radius, :, 0] = data_flat[src_idx]
        x_np = x_np.reshape(T, window_size * P, 1)  # (T, W*P, 1)
        print(f"  [ChebGCN] sliding_window mode: {graph_P} nodes/window (P={P}, W={window_size}), "
              f"{window_edges_np.shape[1]} edges, {T} windows")

    else:
        # Mode 0: spatial_only (current default)
        n_feat = 1
        x_np = data_3d.reshape(T, P, 1).astype(np.float32)
        x_target_np = x_np.copy()
        edge_index_t = torch.as_tensor(spatial_edges_np, dtype=torch.long, device=device)
        graph_P = P
        print(f"  [ChebGCN] spatial_only mode: {P} nodes, {spatial_edges_np.shape[1]} edges, {T} timepoints as batch")

    # ---- Model ----
    model = ChebRefineModel(n_feat, config.cheb_hidden_dim, config.cheb_out_channels, config.cheb_K).to(device)
    decoder = nn.Linear(config.cheb_out_channels, 1).to(device)

    # Warmstart
    cheb_epochs = config.cheb_epochs
    use_warmstart = getattr(config, 'use_cheb_warmstart', True)
    if use_warmstart and cheb_warmstart is not None and 'model_state' in cheb_warmstart:
        try:
            model.load_state_dict(cheb_warmstart['model_state'])
            decoder.load_state_dict(cheb_warmstart['decoder_state'])
            cheb_epochs = max(10, config.cheb_epochs // 3)
            print(f"  [ChebGCN] Warmstart from previous iteration, fine-tuning {cheb_epochs} epochs")
        except Exception:
            print(f"  [ChebGCN] Warmstart failed (shape mismatch), training from scratch")

    opt = AdamW(list(model.parameters()) + list(decoder.parameters()), lr=config.cheb_lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    use_amp = getattr(config, 'use_amp', True) and device.type == 'cuda'
    cheb_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training: mini-batch over timepoints
    cheb_batch_size = getattr(config, 'cheb_batch_size', 256)

    model.train(); decoder.train()
    for ep in range(cheb_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = np.random.permutation(T)
        for b_start in range(0, T, cheb_batch_size):
            b_idx = perm[b_start:b_start + cheb_batch_size]
            B = len(b_idx)
            # Stack B graph instances
            x_batch = torch.as_tensor(
                x_np[b_idx].reshape(B * graph_P, n_feat), dtype=torch.float32, device=device)
            # Replicate edge_index for B graphs
            offsets = torch.arange(B, device=device).unsqueeze(1) * graph_P
            ei_batch = (edge_index_t.unsqueeze(0) + offsets.unsqueeze(1)).reshape(2, -1)

            # Target: only center timepoint's P nodes
            if temporal_mode == 'sliding_window':
                # Extract center time-slice from each window
                target_batch = torch.as_tensor(
                    x_target_np[b_idx].reshape(B * P, 1), dtype=torch.float32, device=device)
            else:
                target_batch = torch.as_tensor(
                    x_target_np[b_idx].reshape(B * P, 1), dtype=torch.float32, device=device)

            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                emb = model(x_batch, ei_batch)
                if temporal_mode == 'sliding_window':
                    # Only decode center time-slice nodes
                    center_offset = temporal_radius * P
                    center_indices = []
                    for b in range(B):
                        start = b * graph_P + center_offset
                        center_indices.extend(range(start, start + P))
                    emb_center = emb[center_indices]
                    dec = decoder(emb_center)
                else:
                    dec = decoder(emb)
                loss = loss_fn(dec, target_batch)
            cheb_scaler.scale(loss).backward()
            cheb_scaler.step(opt)
            cheb_scaler.update()
            epoch_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 10 == 0:
            print(f"[ChebGCN] Epoch {ep + 1}/{cheb_epochs} | MSE: {epoch_loss / max(n_batches, 1):.6f}")

    # Cache trained model
    if cheb_warmstart is not None:
        cheb_warmstart['model_state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        cheb_warmstart['decoder_state'] = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

    # ---- Inference ----
    model.eval(); decoder.eval()
    refined_all = np.empty((T, P), dtype=np.float32)
    with torch.no_grad():
        for b_start in range(0, T, cheb_batch_size):
            b_end = min(b_start + cheb_batch_size, T)
            B = b_end - b_start
            x_batch = torch.as_tensor(
                x_np[b_start:b_end].reshape(B * graph_P, n_feat), dtype=torch.float32, device=device)
            offsets = torch.arange(B, device=device).unsqueeze(1) * graph_P
            ei_batch = (edge_index_t.unsqueeze(0) + offsets.unsqueeze(1)).reshape(2, -1)

            emb = model(x_batch, ei_batch)
            if temporal_mode == 'sliding_window':
                center_offset = temporal_radius * P
                center_indices = []
                for b in range(B):
                    start = b * graph_P + center_offset
                    center_indices.extend(range(start, start + P))
                dec = decoder(emb[center_indices])
            else:
                dec = decoder(emb)
            refined_all[b_start:b_end] = dec.cpu().numpy().reshape(B, P)

    refined_3d = refined_all.reshape(T, H, W)
    return refined_3d

class EarlyStopper:
    def __init__(self, patience=30, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
        self.state = None

    def step(self, value, model):
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            self.state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.count += 1
            return self.count >= self.patience

def extract_features_block(model, X_block_t, layer_name=None):
    """
    X_block_t: torch tensor (B, C, H, W)
    If layer_name is None: return None to signal "use input"
    Else: return numpy (B, F) features from specified layer.
    """
    if layer_name is None:
        return None
    layer_map = {'Dense1': 1, 'Dense2': 2, 'Dense3': 3}
    layer_num = layer_map.get(layer_name, 1)
    with torch.no_grad():
        feats = model.get_layer_output(X_block_t, layer_num)  # must be supported by your model
    feats = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
    return feats

def _compute_single_P(args):
    """Worker for parallel P computation."""
    key, feats, B_center, HD_type, perplexity = args
    P_list = calculate_P_improved(feats, B_center, HD_type, perplexity, offset=0, include_tail=True)
    if P_list:
        return key, P_list[0]
    return key, None

def _precompute_center_P_cache(X_train_auto, batch_size, context, stride, HD_type, perplexity, config=None):
    from concurrent.futures import ThreadPoolExecutor
    n = X_train_auto.shape[0]

    # Collect all possible offsets that get_epoch_offset() can produce
    offsets = {0}
    if config is not None and getattr(config, 'shift_enabled', False):
        for epoch in range(config.max_epochs):
            offsets.add(get_epoch_offset(epoch, config))

    # Collect unique (key, feats) pairs
    tasks = {}
    for offset in sorted(offsets):
        i = max(0, offset - context)
        if i >= n - 1:
            i = 0
        while i < n:
            end = min(i + batch_size + 2 * context, n)
            center_start = min(i + context, end)
            center_end = min(center_start + batch_size, end)
            B_center = center_end - center_start
            if B_center >= 2:
                key = (center_start, B_center)
                if key not in tasks:
                    feats = X_train_auto[center_start:center_start + B_center].reshape(B_center, -1)
                    tasks[key] = (key, feats, B_center, HD_type, perplexity)
            i += stride

    # Parallel P computation (numba releases GIL, so threads work well)
    cache = {}
    n_workers = min(8, len(tasks))
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for key, P in pool.map(_compute_single_P, tasks.values()):
                if P is not None:
                    cache[key] = P
    else:
        for args in tasks.values():
            key, P = _compute_single_P(args)
            if P is not None:
                cache[key] = P

    print(f"  [P-cache] {len(cache)} entries for {len(offsets)} offsets (parallel={n_workers})")
    return cache

def train_model_overlap_temporal(model, X_train_auto, out_model_name, config, device, layer_name=None, stage_num=1):
    """
    Overlap windows with context:
      - forward on [start:end] full window
      - compute KL only on center slice vs P_center (built on center features)
      - add ACF-weighted temporal smoothness on Y_full
    Uses AMP (Automatic Mixed Precision) for ~1.3-1.5x training speedup.
    """
    model.to(device)

    # torch.compile: disabled by default — overhead exceeds benefit for small conv models.
    # Enable with config.use_torch_compile = True for larger models.
    use_compile = getattr(config, 'use_torch_compile', False) and hasattr(torch, 'compile')
    _compiled_model = None
    if use_compile and isinstance(model, nn.Module) and not hasattr(model, '_orig_mod'):
        try:
            _compiled_model = torch.compile(model, mode='default')
        except Exception:
            _compiled_model = None
    fwd_model = _compiled_model if _compiled_model is not None else model

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = create_kl_divergence_stable(config.low_dim)

    # AMP setup for mixed precision training
    use_amp = getattr(config, 'use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # data & sizes
    n = X_train_auto.shape[0]
    if config.stride is None or config.stride <= 0:
        config.stride = max(1, config.batch_size // 2)
    context = max(0, int(config.context))
    batch_size = int(config.batch_size)

    # tensors
    X_all_t = torch.as_tensor(X_train_auto, dtype=torch.float32, device=device)

    # ACF weights from raw input (n, d)
    X_flat = X_train_auto.reshape(n, -1)
    acf_w = make_acf_weights(X_flat, max_lag=config.temporal_max_lag, smooth_window=config.smooth_window).to(device)

    # Boundary-aware temporal loss: precompute weights from high-D data
    # event_boundary_min_stage: only apply boundary loss from this stage onward (default=1 = all stages)
    bw_all_t = None
    min_stage = getattr(config, 'event_boundary_min_stage', 1)
    if getattr(config, 'use_event_boundary_loss', False) and stage_num >= min_stage:
        bw_np = compute_boundary_weights(
            X_flat,
            smooth_window=3,
            percentile=getattr(config, 'event_boundary_percentile', 75)
        )
        bw_all_t = torch.as_tensor(bw_np, dtype=torch.float32, device=device)
        print(f"  [Boundary weights] stage={stage_num} shape={bw_all_t.shape}, mean={bw_all_t.mean():.3f}, min={bw_all_t.min():.3f}")
    elif getattr(config, 'use_event_boundary_loss', False):
        print(f"  [Boundary loss] skipped for stage {stage_num} (min_stage={min_stage})")

    # Temporal distance correlation loss: precompute high-D consecutive distances
    d_high_all_t = None
    if getattr(config, 'use_tdist_loss', False) or getattr(config, 'use_ratio_loss', False):
        d_high_all = np.sqrt(np.sum((X_flat[1:] - X_flat[:-1]) ** 2, axis=1)).astype(np.float32)
        d_high_all_t = torch.as_tensor(d_high_all, dtype=torch.float32, device=device)
        print(f"  [TDist] precomputed d_high: shape={d_high_all_t.shape}, mean={d_high_all_t.mean():.3f}, std={d_high_all_t.std():.3f}")

    stopper = EarlyStopper(patience=config.patience)
    best_path = out_model_name

    # Stage-1: precompute identical P_center cache (pre-convert to GPU tensors)
    P_center_cache = None
    if layer_name is None:
        P_center_cache_np = _precompute_center_P_cache(
            X_train_auto, batch_size, context, config.stride, config.HD_type, config.perplexity, config=config
        )
        # Pre-convert all P matrices to GPU tensors to avoid repeated CPU→GPU transfers
        P_center_cache = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in P_center_cache_np.items()
        }

    # Dynamic P cache for later stages — recompute every p_update_frequency epochs
    _dynamic_P_cache = {}
    _dynamic_P_epoch = -1

    for epoch in range(config.max_epochs):
        model.train()  # sets training mode on original model (fwd_model shares weights)
        epoch_loss = 0.0
        steps = 0

        # epoch offset to vary coverage (kept small to not break continuity)
        current_offset = get_epoch_offset(epoch, config)
        i = max(0, current_offset - context)
        # avoid empty epochs when offset goes too far
        if i >= n - 1:
            i = 0

        while i < n:
            start = i
            end = min(i + batch_size + 2 * context, n)
            center_start = min(i + context, end)
            center_end = min(center_start + batch_size, end)

            B_full = end - start
            B_center = center_end - center_start
            if B_center < 2 or B_full < 2:
                break

            # Prepare inputs
            X_full = X_all_t[start:end].permute(0, 3, 1, 2)  # (B_full, C, H, W)

            # Build P on center features (input or layer) — outside AMP context
            if layer_name is None and P_center_cache is not None:
                P_center = P_center_cache.get((center_start, B_center))
                if P_center is None:
                    i += config.stride
                    continue
                # P_center is already a GPU tensor (pre-converted)
            else:
                # Dynamic P caching for later stages: recompute every p_update_frequency epochs
                cache_key = (center_start, B_center)
                p_freq = getattr(config, 'p_update_frequency', 20)
                need_recompute = (epoch - _dynamic_P_epoch >= p_freq) or (cache_key not in _dynamic_P_cache)

                if need_recompute:
                    if layer_name is None:
                        X_center_feat = X_train_auto[center_start:center_end].reshape(B_center, -1)
                    else:
                        with torch.no_grad():
                            feats_full = extract_features_block(model, X_full, layer_name)
                        start_in_full = context
                        end_in_full = context + B_center
                        X_center_feat = feats_full[start_in_full:end_in_full, :]

                    P_list_center = calculate_P_improved(
                        X_center_feat, B_center, config.HD_type, config.perplexity,
                        offset=0, include_tail=True
                    )
                    if len(P_list_center) == 0:
                        i += config.stride
                        continue
                    _dynamic_P_cache[cache_key] = P_list_center[0]
                    _dynamic_P_epoch = epoch

                P_center = torch.as_tensor(_dynamic_P_cache[cache_key], dtype=torch.float32, device=device)

            # Forward + Loss under AMP autocast
            with torch.amp.autocast('cuda', enabled=use_amp):
                Y_full = fwd_model(X_full)                       # (B_full, low_dim)

                # Slice center embeddings from Y_full
                Y_center = Y_full[context:context + B_center]

                # Losses
                L_kl = criterion(P_center, Y_center)

                # Temporal loss mode:
                #   'add' (default): L_temporal + L_event (original)
                #   'replace': use L_event instead of L_temporal
                #   'contrast': L_temporal + boundary_contrast_loss
                #   'replace_contrast': boundary_contrast_loss only (no L_temporal)
                event_mode = getattr(config, 'event_loss_mode', 'add')
                _lt = getattr(config, 'lambda_temporal', 0.2)

                if event_mode == 'replace' and bw_all_t is not None:
                    # Boundary-aware temporal loss REPLACES uniform smoothing
                    bw_start = max(start, 0)
                    bw_end = min(end - 1, bw_all_t.shape[0])
                    if bw_end > bw_start:
                        bw_slice = bw_all_t[bw_start:bw_end]
                        L_event = boundary_aware_temporal_loss(
                            Y_full, bw_slice,
                            max_lag=getattr(config, 'event_max_lag', 20)
                        )
                        loss = L_kl + _lt * L_event
                    else:
                        loss = L_kl + _lt * _temporal_loss(Y_full, acf_w, config)
                elif event_mode == 'contrast' and bw_all_t is not None:
                    # Uniform temporal + contrastive boundary loss
                    L_temporal = _temporal_loss(Y_full, acf_w, config)
                    loss = L_kl + _lt * L_temporal
                    bw_start = max(start, 0)
                    bw_end = min(end - 1, bw_all_t.shape[0])
                    if bw_end > bw_start:
                        bw_slice = bw_all_t[bw_start:bw_end]
                        L_contrast = boundary_contrast_loss(
                            Y_full, bw_slice,
                            margin=getattr(config, 'contrast_margin', 1.0)
                        )
                        loss = loss + config.lambda_event * L_contrast
                elif event_mode == 'replace_contrast' and bw_all_t is not None:
                    # Contrastive boundary loss REPLACES uniform smoothing
                    bw_start = max(start, 0)
                    bw_end = min(end - 1, bw_all_t.shape[0])
                    if bw_end > bw_start:
                        bw_slice = bw_all_t[bw_start:bw_end]
                        L_contrast = boundary_contrast_loss(
                            Y_full, bw_slice,
                            margin=getattr(config, 'contrast_margin', 1.0)
                        )
                        loss = L_kl + _lt * L_contrast
                    else:
                        loss = L_kl + _lt * _temporal_loss(Y_full, acf_w, config)
                else:
                    # Default 'add' mode: original behavior
                    L_temporal = _temporal_loss(Y_full, acf_w, config)
                    loss = L_kl + _lt * L_temporal
                    if bw_all_t is not None:
                        bw_start = max(start, 0)
                        bw_end = min(end - 1, bw_all_t.shape[0])
                        if bw_end > bw_start:
                            bw_slice = bw_all_t[bw_start:bw_end]
                            L_event = boundary_aware_temporal_loss(
                                Y_full, bw_slice,
                                max_lag=getattr(config, 'event_max_lag', 20)
                            )
                            loss = loss + config.lambda_event * L_event

                # Temporal distance correlation / ratio losses (appended to ALL modes)
                if d_high_all_t is not None and epoch >= getattr(config, 'tdist_warmup_epochs', 10):
                    d_high_slice = d_high_all_t[start:start + Y_full.shape[0] - 1]
                    if d_high_slice.shape[0] >= 3:
                        if getattr(config, 'use_tdist_loss', False):
                            L_tdist = temporal_distance_correlation_loss(Y_full, d_high_slice)
                            loss = loss + config.lambda_tdist * L_tdist
                        if getattr(config, 'use_ratio_loss', False):
                            L_ratio = distance_ratio_preservation_loss(Y_full, d_high_slice)
                            loss = loss + config.lambda_ratio * L_ratio

                # Directional consistency loss (anti-zigzag, appended to ALL modes)
                if getattr(config, 'use_directional_loss', False):
                    L_dir = directional_consistency_loss(Y_full)
                    loss = loss + config.lambda_directional * L_dir

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            steps += 1

            i += config.stride

        if steps == 0:
            print(f"[Warn] No valid steps in epoch {epoch+1}")
            continue

        scheduler.step()
        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch+1}/{config.max_epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f} | offset={current_offset}")

        # Early stopping
        if stopper.step(avg_loss, model):
            print(f"Early stop at epoch {epoch+1} (best {stopper.best:.6f})")
            break
        else:
            # Save best so far
            if best_path is not None:
                torch.save(model.state_dict(), best_path)

    # Load best weights (from stopper) if available
    if stopper.state is not None:
        model.load_state_dict(stopper.state)
    elif best_path is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval()
    model._best_loss = stopper.best  # attach best loss for retry logic
    return model

def _reinit_model_weights(model):
    """Re-initialize all learnable parameters in-place."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def do_multi_stage_dr_training_overlap(dr_model, X_train_proj, config, outer_iter,
                                       subject_num, model_save_dir, device):
    saved_paths = []
    # Stage 1: Input space — with retry mechanism for stability
    mpath1 = os.path.join(model_save_dir, f"sub-{subject_num:02d}_outer{outer_iter}_stage1_biopd.pth")

    stage1_loss_threshold = getattr(config, 'stage1_loss_threshold', 0.7)
    max_retries = getattr(config, 'stage1_max_retries', 3)

    best_across_retries = float('inf')
    best_state_across = None
    for attempt in range(1, max_retries + 1):
        dr_model = train_model_overlap_temporal(
            dr_model, X_train_proj, mpath1, config, device, layer_name=None, stage_num=1
        )
        best_loss = getattr(dr_model, '_best_loss', 0.0)

        # Track best across all retry attempts
        if best_loss < best_across_retries:
            best_across_retries = best_loss
            best_state_across = {k: v.cpu().clone() for k, v in dr_model.state_dict().items()}

        if best_loss <= stage1_loss_threshold:
            if attempt > 1:
                print(f"  [Retry] Stage 1 succeeded on attempt {attempt} (loss={best_loss:.4f})")
            break
        if attempt < max_retries:
            # Smart retry: skip if no meaningful improvement between attempts
            improvement = (best_across_retries - best_loss) / max(best_across_retries, 1e-8)
            if attempt >= 2 and improvement < 0.05:
                print(f"  [Retry] Stage 1 loss={best_loss:.4f}, no improvement vs best={best_across_retries:.4f}, stopping retries")
                break
            print(f"  [Retry] Stage 1 loss={best_loss:.4f} > {stage1_loss_threshold} on attempt {attempt}/{max_retries}, reinitializing...")
            _reinit_model_weights(dr_model)
        else:
            print(f"  [Retry] Stage 1 loss={best_loss:.4f} after {max_retries} attempts, proceeding with best result")

    # Restore best state across all retry attempts
    if best_state_across is not None:
        dr_model.load_state_dict(best_state_across)
        dr_model._best_loss = best_across_retries

    saved_paths.append(mpath1)
    # Save stage 1 best model
    torch.save(dr_model.state_dict(), mpath1)

    # Progressive stages — model already in memory, skip redundant disk loads
    if config.n_recur > 1:
        mpath2 = os.path.join(model_save_dir, f"sub-{subject_num:02d}_outer{outer_iter}_stage2_biopd.pth")
        dr_model = train_model_overlap_temporal(
            dr_model, X_train_proj, mpath2, config, device, layer_name='Dense1', stage_num=2
        )
        saved_paths.append(mpath2)

    if config.n_recur > 2:
        mpath3 = os.path.join(model_save_dir, f"sub-{subject_num:02d}_outer{outer_iter}_stage3_biopd.pth")
        dr_model = train_model_overlap_temporal(
            dr_model, X_train_proj, mpath3, config, device, layer_name='Dense2', stage_num=3
        )
        saved_paths.append(mpath3)

    if config.n_recur > 3:
        mpath4 = os.path.join(model_save_dir, f"sub-{subject_num:02d}_outer{outer_iter}_stage4_biopd.pth")
        dr_model = train_model_overlap_temporal(
            dr_model, X_train_proj, mpath4, config, device, layer_name='Dense3', stage_num=4
        )
        saved_paths.append(mpath4)

    return dr_model, saved_paths

def co_train_refine_DR_overlap(raw_data, dr_model, config, subject_num, model_save_dir, device,
                               gw_proj_mat=None):
    os.makedirs(model_save_dir, exist_ok=True)
    data_cur = raw_data.copy()
    all_saved = []

    # ChebGCN caching: save trained model from iteration 1, fine-tune in iteration 2+
    _cheb_cache = {}

    for outer in range(1, config.n_iterations + 1):
        print(f"\n=== [{config.run_name}] Outer Iteration {outer}/{config.n_iterations} ===")
        # 1) ChebGCN refinement (GW only for first outer iter on original voxels)
        cur_gw = gw_proj_mat if outer == 1 else None
        # Pass _cheb_cache always: iteration 1 saves weights, iteration 2+ loads & fine-tunes
        refined_3d = refine_data_with_ChebGCN(data_cur, config, device, gw_proj_mat=cur_gw,
                                                cheb_warmstart=_cheb_cache if outer > 1 else _cheb_cache)

        # Save refined_3d for reproducibility
        refined_3d_path = os.path.join(model_save_dir, f"sub-{subject_num:02d}_outer{outer}_refined3d.npy")
        np.save(refined_3d_path, refined_3d)

        if refined_3d.ndim == 3:
            T, H, W = refined_3d.shape
            refined_2d = refined_3d.reshape(T, H * W)
            X_train_proj = refined_3d[..., np.newaxis]        # (T, H, W, 1)
            input_shape = (H, W)
        else:
            T2, S2 = refined_3d.shape
            refined_2d = refined_3d
            X_train_proj = refined_2d.reshape(T2, S2, 1, 1)   # (T2, S2, 1, 1)
            input_shape = (S2, 1)

        # 2) Multi-stage DR (overlap-temporal)
        dr_model, saved_paths = do_multi_stage_dr_training_overlap(
            dr_model, X_train_proj, config, outer, subject_num, model_save_dir, device
        )
        all_saved.extend(saved_paths)

        # 3) Update for next outer loop
        data_cur = refined_2d

    return dr_model, refined_3d, input_shape, all_saved

def test_knn_for_ks(emb, labels, k_list, cv_folds=10):
    scores = {}
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        sc = cross_val_score(knn, emb, labels, cv=cv_folds).mean()
        scores[k] = sc
    return scores

def extract_and_plot_embeddings(saved_model_paths, input_shape, data_final_refined_3d,
                                labels, config, subject_num, roi_name, plot_save_dir, device):
    embeddings = []
    colors = [
        'darkorange', 'deepskyblue', 'gold', 'hotpink', 'lime', 'k', 'darkviolet', 'peru',
        'mediumblue', 'olive', 'midnightblue', 'palevioletred', 'c', 'y', 'b', 'tan', 'navy',
        'plum', 'slategray', 'lightseagreen', 'purple', 'lightcoral', 'red', 'skyblue', 'moccasin',
        'darkorchid', 'indigo', 'palegreen', 'crimson', 'm', 'steelblue', 'darkgoldenrod',
        'burlywood', 'fuchsia', 'dodgerblue', 'greenyellow', 'khaki', 'lavender', 'azure'
    ]
    colormap = ListedColormap(colors[::-1])

    if data_final_refined_3d.ndim == 3:
        X_proj = data_final_refined_3d[..., np.newaxis]
        X_t = torch.as_tensor(X_proj, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    else:
        n = data_final_refined_3d.shape[0]
        X_proj = data_final_refined_3d.reshape(n, -1, 1, 1)
        X_t = torch.as_tensor(X_proj, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    for mpth in saved_model_paths:
        if not os.path.exists(mpth):
            print(f"[Warn] Missing model: {mpth}")
            continue
        tmp = create_model(
            input_shape=input_shape,
            num_conv_layers=config.num_conv_layers,
            filters_list=config.filters_list,
            kernel_size=config.kernel_size,
            alpha=config.alpha,
            dense_units=config.dense_units,
            final_units=config.final_units
        )
        tmp.load_state_dict(torch.load(mpth, map_location=device), strict=True)
        tmp.to(device).eval()
        with torch.no_grad():
            emb = tmp(X_t).cpu().numpy()
        emb = post_smooth_embedding(emb, getattr(config, 'post_smooth_sigma', 3))
        embeddings.append((mpth, emb))
        del tmp
        torch.cuda.empty_cache()

    os.makedirs(plot_save_dir, exist_ok=True)
    for idx, (mpth, emb) in enumerate(embeddings, start=1):
        plt.figure(figsize=config.figure_size)
        plt.rcParams.update({'font.size': config.font_size})
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=colormap, marker='o', s=config.marker_size)
        plt.xlabel('Bio-PD 1'); plt.ylabel('Bio-PD 2')
        plt.colorbar(sc); plt.tight_layout()
        out_fig = os.path.join(plot_save_dir, f"model_{idx}_sub-{subject_num:02d}_{roi_name}.png")
        plt.savefig(out_fig); plt.close()
        print(f"Saved plot: {out_fig}")
    return embeddings

def evaluate_embeddings(embeddings, labels, subject_num, roi_name, config, save_dir):
    rows = []
    for idx, (mpth, emb) in enumerate(embeddings, start=1):
        scores = test_knn_for_ks(emb, labels, config.knn_k_values, config.cv_folds)
        rows.append({
            "subject_number": subject_num,
            "ROI": roi_name,
            "model": f"model_{idx}",
            "knn_cv_mean": float(np.mean(list(scores.values()))),
            "cv_scores": scores
        })
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "evaluation_metrics.csv")
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    print(f"Saved evaluation metrics: {csv_path}")
    return rows

def process_subject_roi_overlap(subject_num, roi_number, config, device):
    roi_info = config.roi_info[roi_number]
    roi_name = roi_info["name"]
    model_dir_name = roi_info["model_dir"]
    data_file = roi_info["data_file"]

    print(f"\n{'#'*60}")
    print(f"[{config.run_name}] Subject {subject_num:02d}, ROI {roi_number} ({roi_name})")
    print(f"{'#'*60}")

    # Apply ROI-adaptive parameters
    if config.use_roi_adaptive and roi_name in config.roi_adaptive_params:
        params = config.roi_adaptive_params[roi_name]
        for k, v in params.items():
            setattr(config, k, v)
        print(f"  [ROI-adaptive] {params}")

    data_path = os.path.join(config.base_data_path, roi_name, f"sub-{subject_num:02d}_{data_file}")
    if not os.path.exists(data_path):
        print(f"[Skip] Missing data: {data_path}")
        return None

    data = np.load(data_path)
    X_train, batch_size, n = sherlock_para(data, config.balance_degree)
    print(f"X_train shape: {X_train.shape} | batch_size: {batch_size} | n: {n}")
    config.batch_size = batch_size
    if config.stride is None or config.stride <= 0:
        config.stride = max(1, batch_size // 2)

    _, S = X_train.shape
    H = int(math.floor(math.sqrt(S)))
    W = int(math.ceil(S / H))
    input_shape = (H, W)
    print(f"Input shape for DR: {input_shape}")

    # Compute GW projection matrix once per ROI and subject.
    gw_proj_mat = None
    if config.use_gw_layout:
        print(f"  [GW] Computing optimal-transport spatial projection ({S} voxels -> {H}x{W} grid)...")
        # Apply TimeCORR first for cleaner correlations
        import statsmodels.api as sm_acf
        n_samples = X_train.shape[0]
        A_feat = np.empty((n_samples, S))
        for f in range(S):
            A_feat[:, f] = sm_acf.tsa.acf(X_train[:, f], fft=False, nlags=n_samples - 1, missing='drop')
        A_mean = np.nanmean(A_feat, axis=1)
        acf = np.convolve(A_mean, np.ones(config.smooth_window), 'same') / config.smooth_window
        drop_idx = np.where(acf < 0)[0]
        dropoff = n_samples if len(drop_idx) == 0 else drop_idx[0]
        # Vectorized M_corr construction (replaces O(n²) Python loop)
        row_idx = np.arange(n_samples)
        lag_matrix = np.abs(row_idx[:, None] - row_idx[None, :])
        M_corr = np.zeros((n_samples, n_samples), dtype=np.float32)
        valid = (lag_matrix > 0) & (lag_matrix < dropoff)
        M_corr[valid] = acf[lag_matrix[valid]]
        row_sums = M_corr.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        M_corr /= row_sums
        X_smoothed = M_corr @ X_train
        gw_proj_mat = compute_gw_projection(X_smoothed, H, W)
        print(f"  [GW] Projection computed. Shape: {gw_proj_mat.shape}")

    model_save_dir = os.path.join(config.base_model_save_dir, model_dir_name)
    os.makedirs(model_save_dir, exist_ok=True)

    dr_model = create_model(
        input_shape=input_shape,
        num_conv_layers=config.num_conv_layers,
        filters_list=config.filters_list,
        kernel_size=config.kernel_size,
        alpha=config.alpha,
        dense_units=config.dense_units,
        final_units=config.final_units
    )

    dr_model, data_final_refined_3d, final_input_shape, saved_paths = co_train_refine_DR_overlap(
        raw_data=X_train, dr_model=dr_model, config=config,
        subject_num=subject_num, model_save_dir=model_save_dir, device=device,
        gw_proj_mat=gw_proj_mat
    )

    print("Saved models:")
    for p in saved_paths:
        print("  ", p)

    # Ensure final model is saved to disk
    final_save_path = os.path.join(model_save_dir, f"sub-{subject_num:02d}_final.pth")
    torch.save(dr_model.state_dict(), final_save_path)
    if final_save_path not in saved_paths:
        saved_paths.append(final_save_path)

    return {
        'subject_num': subject_num,
        'roi_number': roi_number,
        'roi_name': roi_name,
        'n': n,
        'input_shape': final_input_shape,
        'data_final_refined_3d': data_final_refined_3d,
        'saved_model_paths': saved_paths,
        'dr_model': dr_model,
    }

# ============================================================================
# Command-line interface
# ============================================================================
ANALYSIS_MODE_MAP = {
    "standard": {
        "use_gw_layout": False,
        "n_iterations": 2,
        "use_roi_adaptive": False,
    },
    "gw_layout": {
        "use_gw_layout": True,
        "n_iterations": 2,
        "use_roi_adaptive": False,
    },
    "iterative_refinement": {
        "use_gw_layout": False,
        "n_iterations": 3,
        "use_roi_adaptive": False,
    },
    "roi_adaptive": {
        "use_gw_layout": False,
        "n_iterations": 2,
        "use_roi_adaptive": True,
    },
    "gw_iterative": {
        "use_gw_layout": True,
        "n_iterations": 3,
        "use_roi_adaptive": False,
    },
    "gw_roi_adaptive": {
        "use_gw_layout": True,
        "n_iterations": 2,
        "use_roi_adaptive": True,
    },
    "iterative_roi_adaptive": {
        "use_gw_layout": False,
        "n_iterations": 3,
        "use_roi_adaptive": True,
    },
    "full": {
        "use_gw_layout": True,
        "n_iterations": 3,
        "use_roi_adaptive": True,
    },
}


def _parse_roi_indices(value, n_rois):
    if value is None or value.lower() == "all":
        return list(range(n_rois))
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= n_rois:
            raise ValueError(f"ROI index {idx} is outside the valid range 0-{n_rois - 1}.")
        out.append(idx)
    return out


def _load_labels(labels_path, label_column):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    sheet = pd.read_csv(labels_path, encoding="utf-8")
    if isinstance(label_column, str) and not label_column.isdigit():
        if label_column not in sheet.columns:
            raise KeyError(f"Column '{label_column}' not found in label file.")
        labels = sheet[label_column].to_numpy()
    else:
        idx = int(label_column)
        labels = sheet.iloc[:, idx].to_numpy()
    return labels.astype(int)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Bio-PD fMRI workflow."
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="full",
        choices=list(ANALYSIS_MODE_MAP.keys()),
        help="Analysis mode to run."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML configuration file.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id exposed to this run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing ROI-specific fMRI arrays.")
    parser.add_argument("--labels_path", type=str, default=None, help="CSV file containing scene labels.")
    parser.add_argument("--label_column", type=str, default="9", help="Label column name or zero-based column index.")
    parser.add_argument("--out_dir", type=str, default="./results/biopd_fmri", help="Output directory.")
    parser.add_argument("--subject_start", type=int, default=None, help="First subject index, inclusive.")
    parser.add_argument("--subject_end", type=int, default=None, help="Last subject index, exclusive.")
    parser.add_argument("--rois", type=str, default="all", help="Comma-separated ROI indices, or 'all'.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override maximum training epochs.")
    parser.add_argument("--n_iterations", type=int, default=None, help="Override outer refinement iterations.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Run training without KNN evaluation.")
    args = parser.parse_args()

    config = Config(args.config)
    config.cuda_device = args.gpu
    config.seed = args.seed

    analysis_cfg = ANALYSIS_MODE_MAP[args.analysis]
    config.use_gw_layout = analysis_cfg["use_gw_layout"]
    config.n_iterations = analysis_cfg["n_iterations"]
    config.use_roi_adaptive = analysis_cfg["use_roi_adaptive"]
    config.run_name = f"{args.analysis}_seed{args.seed}"

    if args.n_iterations is not None:
        config.n_iterations = args.n_iterations
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.data_dir is not None:
        config.base_data_path = args.data_dir
    if args.labels_path is not None:
        config.labels_path = args.labels_path
    if args.subject_start is not None:
        config.subject_start = args.subject_start
    if args.subject_end is not None:
        config.subject_end = args.subject_end

    run_out_dir = os.path.join(args.out_dir, config.run_name)
    config.base_model_save_dir = run_out_dir
    config.base_plot_save_dir = run_out_dir
    os.makedirs(run_out_dir, exist_ok=True)

    device = setup_environment(config)
    all_labels = _load_labels(config.labels_path, args.label_column)
    roi_indices = _parse_roi_indices(args.rois, len(config.roi_info))

    results_summary = []
    for subject_num in range(config.subject_start, config.subject_end):
        for roi_number in roi_indices:
            config.stride = None

            if config.use_roi_adaptive:
                config.lambda_temporal = 0.2
                config.perplexity = 30
                config.context = 64
                config.patience = 30
                config.cheb_K = 3
                config.top_k_correlations = 8

            result = process_subject_roi_overlap(subject_num, roi_number, config, device)
            if result is None:
                continue

            n = result["n"]
            input_shape = result["input_shape"]
            data_final_refined_3d = result["data_final_refined_3d"]
            saved_model_paths = result["saved_model_paths"]
            roi_name = result["roi_name"]
            labels = all_labels[:n]

            if args.skip_evaluation:
                results_summary.append({"roi": roi_name, "mean_knn": np.nan, "per_k": {}})
                continue

            final_model_path = saved_model_paths[-1]
            tmp_model = create_model(
                input_shape=input_shape,
                num_conv_layers=config.num_conv_layers,
                filters_list=config.filters_list,
                kernel_size=config.kernel_size,
                alpha=config.alpha,
                dense_units=config.dense_units,
                final_units=config.final_units,
            )
            tmp_model.load_state_dict(torch.load(final_model_path, map_location=device))
            tmp_model.to(device).eval()

            if data_final_refined_3d.ndim == 3:
                X_eval = data_final_refined_3d[..., np.newaxis]
            else:
                T2, S2 = data_final_refined_3d.shape
                X_eval = data_final_refined_3d.reshape(T2, S2, 1, 1)
            X_eval_t = torch.as_tensor(X_eval, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

            with torch.no_grad():
                emb = tmp_model(X_eval_t).cpu().numpy()
            emb = post_smooth_embedding(emb, getattr(config, "post_smooth_sigma", 3))

            emb_dir = os.path.join(run_out_dir, "embeddings")
            os.makedirs(emb_dir, exist_ok=True)
            np.save(os.path.join(emb_dir, f"sub-{subject_num:02d}_{roi_name}_biopd_embedding.npy"), emb)

            knn_scores = {}
            for k in config.knn_k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                score = cross_val_score(knn, emb, labels, cv=config.cv_folds).mean()
                knn_scores[k] = score
            mean_knn = float(np.mean(list(knn_scores.values())))

            print(f"\n[{config.run_name}] {roi_name} final KNN mean = {mean_knn:.4f}")
            print(f"  Per-k: {knn_scores}")
            results_summary.append({"roi": roi_name, "mean_knn": mean_knn, "per_k": knn_scores})
            torch.cuda.empty_cache()

    summary_path = os.path.join(run_out_dir, "summary_metrics.csv")
    pd.DataFrame(results_summary).to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"[{config.run_name}] Summary")
    print(f"{'='*60}")
    for row in results_summary:
        print(f"  {row['roi']}: KNN mean = {row['mean_knn']:.4f}")
    print(f"\nSaved summary: {summary_path}")
    print("Bio-PD fMRI workflow completed.")


if __name__ == "__main__":
    main()
