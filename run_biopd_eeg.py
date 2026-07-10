#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bio-PD EEG pipeline for CHB-MIT seizure-risk trajectory analysis.

This standalone script integrates the original project-local modules into a
single publication-ready entry point. The computational pathway is intentionally
kept close to the original implementation: cross-subject ChebGCN pretraining,
per-subject unsupervised trajectory generation, and supervised warning
evaluation. State labels are not used during trajectory generation.

Example:
  python run_biopd_eeg.py --data_root ./data_chb --output_dir ./results/biopd_eeg --gpu 0 --seed 42
"""

import os
import gc
import sys
import re
import math
import glob
import random
import argparse
import warnings
import pickle
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import sklearn
import sklearn.metrics as mpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import mne
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec

from scipy.optimize import fmin_l_bfgs_b
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy, pearsonr, mannwhitneyu, spearmanr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, maximum_filter1d
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

import numba
from numba import njit
import ot
from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend
from torch_geometric.nn import ChebConv

warnings.filterwarnings("ignore")


# ============================================================================
# Optimal-transport utilities
# ============================================================================

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





# ============================================================================
# Gromov-Wasserstein spatial utilities
# ============================================================================

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



# ============================================================================
# Spatial projection utilities
# ============================================================================

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





# ============================================================================
# General preprocessing helpers retained for compatibility
# ============================================================================

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

def rat_data_preprocess(hippo_file_name1, hippo_file_name2, rat_number, balance_degree):
    if rat_number == 0:
        data = np.load(hippo_file_name1)
        labels = np.load(hippo_file_name2)
        n_all, _ = data.shape
        batch_size = 2000  # set 2000 as an upper limit for efficient calculation
        n = batch_size * (n_all // batch_size)
        X_train = data[0:n, :]
        labels = labels[0:n, :]
    else:
        data = np.load(hippo_file_name1)
        labels = np.load(hippo_file_name2)
        if rat_number == 2:
            X_train = data[13747:33747, :]
            labels = labels[13747:33747, :]
        else:
            X_train = data
            labels = labels
        n_all, _ = X_train.shape
        batch_size = n_all // balance_degree
        n = batch_size * balance_degree
        X_train = X_train[0:n, :]
        labels = labels[0:n, :]

    return X_train, labels, n, batch_size

def monkey_data_preprocess(monkey_file_name1, monkey_file_name2):
    data = np.load(monkey_file_name1)
    labels = np.load(monkey_file_name2)

    proj_size = 8  # proj_size = int(np.sqrt(n_HD)) where n, n_HD = data.shape
    colNum = proj_size
    rowNum = proj_size
    avg_vector_proj = []

    for i in range(8):
        mask = labels == i
        selected_trails = np.where(mask)[0]

        concatenated_selected_trails = np.vstack([
            data[trail * 600: (trail + 1) * 600] for trail in selected_trails
        ])

        avg_vector = np.mean(concatenated_selected_trails, axis=0)
        avg_vector = spatiotemporal_projection(avg_vector, rowNum, colNum)

        if i == 0:
            avg_vector_proj = avg_vector
        else:
            avg_vector_proj = np.concatenate((avg_vector_proj, avg_vector), axis=0)

    X_train = []
    for i in range(len(avg_vector_proj) // 600):
        X_train1 = avg_vector_proj[600 * i:600 * (i + 1), :]
        X_train_min = np.min(X_train1)
        X_train_max = np.max(X_train1)
        X_train1 = (X_train1 - X_train_min) / (X_train_max - X_train_min)
        X_train.append(X_train1)

    X_train_proj = np.concatenate(X_train)
    n = X_train_proj.shape[0]
    return X_train_proj, labels, n

def monkey_pos_average(filename3, labels):
    data = np.load(filename3)
    proj_size = 8
    colNum = proj_size
    rowNum = proj_size

    averaged_vectors = []
    for i in range(8):
        mask = labels == i
        selected_trails = np.where(mask)[0]

        concatenated_selected_trails = np.vstack([
            data[trail * 600: (trail + 1) * 600] for trail in selected_trails
        ])

        avg_vector = np.mean(concatenated_selected_trails, axis=0)
        averaged_vectors.append(avg_vector)

    pos_ref = np.vstack(averaged_vectors)
    return pos_ref


# ============================================================================
# Manifold probability utilities
# ============================================================================

@njit
def Hbeta(D, beta):
    P = np.exp(-D * beta)
    # 求和
    sumP = np.float32(0.0)
    for i in range(P.shape[0]):
        sumP += P[i]
    # 加 ε 防止下溢和除零
    sumP = sumP + np.float32(1e-8)

    # 计算 DP
    DP = np.float32(0.0)
    for i in range(P.shape[0]):
        DP += np.float32(D[i] * P[i])

    # 香农熵
    H = np.log(sumP) + beta * DP / sumP

    # 归一化
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


# ============================================================================
# Bio-PD neural network and evaluation helpers
# ============================================================================

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

from scipy.stats import spearmanr

def evaluate_pos_rat_spearman(pos_ref, pred):
    dist_pos = squareform(pdist(pos_ref, metric='euclidean'))
    dist_pred = squareform(pdist(pred, metric='euclidean'))

    rho, pval = spearmanr(dist_pos.flatten(), dist_pred.flatten())
    return rho, pval

import numpy as np
from scipy.spatial.distance import pdist, squareform

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

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

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




# ============================================================================
# Optional single-file CHB-MIT preprocessing helper
# ============================================================================

# ================= Configuration =================
DATA_DIR = "./data_chb/chb01"
OUTPUT_DIR = "./data_chb_preprocessed"
TARGET_FILE = "chb01_03.edf"
SEIZURE_START_SEC = 2996  # seizure onset time (seconds)

TOTAL_DURATION_SEC = 3600  # use 1 hour pre-ictal data
WINDOW_SEC = 2
STEP_SEC = 1

# Feature preprocessing
USE_PCA = True
PCA_VAR_KEEP = 0.95

# GT options: "linear", "sigmoid", "piecewise"
GT_TYPE = "sigmoid"

# Sigmoid GT params
SIGMOID_SHARPNESS = 3.0
SIGMOID_TRANSITION_POINT_SEC = 1200  # 20 min before seizure
# =================================================


class ImprovedEEGPreprocessor:
    """Improved EEG preprocessor to prevent trajectory collapse."""

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Standard 10-20 bipolar channels used in CHB-MIT (may differ by file)
        self.standard_channels = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
        ]

        # Frequency bands (add gamma)
        self.bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }

    @staticmethod
    def compute_spectral_entropy(psd):
        """Spectral entropy of normalized PSD."""
        psd = np.asarray(psd, dtype=np.float64)
        denom = float(np.sum(psd)) + 1e-12
        psd_norm = psd / denom
        return float(scipy_entropy(psd_norm + 1e-12))

    @staticmethod
    def compute_band_ratios(mean_band_powers):
        """Band ratio features: theta/alpha, beta/alpha, delta/theta."""
        ratios = []
        alpha = float(mean_band_powers.get("alpha", 0.0))
        theta = float(mean_band_powers.get("theta", 0.0))
        beta = float(mean_band_powers.get("beta", 0.0))
        delta = float(mean_band_powers.get("delta", 0.0))

        if alpha > 0:
            ratios.append(theta / (alpha + 1e-12))
            ratios.append(beta / (alpha + 1e-12))
        else:
            ratios.extend([0.0, 0.0])

        if theta > 0:
            ratios.append(delta / (theta + 1e-12))
        else:
            ratios.append(0.0)

        return ratios

    @staticmethod
    def compute_connectivity(window_data, n_top=5):
        """
        Simple functional connectivity features from correlation matrix:
        - global mean abs corr
        - max abs corr
        - std corr
        - per-channel mean abs corr for first n_top channels
        """
        x = np.asarray(window_data, dtype=np.float64)
        n_channels = x.shape[0]

        corr = np.corrcoef(x)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        abs_corr = np.abs(corr)
        features = [
            float(np.mean(abs_corr)),
            float(np.max(abs_corr)),
            float(np.std(corr)),
        ]

        ch_conn = np.mean(abs_corr, axis=1)
        topk = int(min(n_top, n_channels))
        features.extend([float(v) for v in ch_conn[:topk]])

        return features

    @staticmethod
    def compute_temporal_dynamics(window_data, max_feats=20):
        """
        Temporal dynamics:
        - per-channel diff std (rate of change proxy)
        - per-channel zero-crossing rate
        Returns up to max_feats total.
        """
        x = np.asarray(window_data, dtype=np.float64)
        n_channels = x.shape[0]
        feats = []

        for ch in range(n_channels):
            sig = x[ch]
            diff = np.diff(sig)
            feats.append(float(np.std(diff)))

            centered = sig - float(np.mean(sig))
            zc = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
            feats.append(float(zc) / float(len(sig) + 1e-12))

        if len(feats) > max_feats:
            feats = feats[:max_feats]
        return feats

    def extract_rich_features(self, window_data, sfreq):
        """Extract band powers + spectral entropy + band ratios + connectivity + temporal dynamics."""
        x = np.asarray(window_data, dtype=np.float64)
        n_channels, n_samples = x.shape

        band_powers_all = {band: [] for band in self.bands}
        spectral_entropies = []

        for ch in range(n_channels):
            freqs, psd = welch(x[ch], fs=sfreq, nperseg=min(256, n_samples))

            for band_name, (fmin, fmax) in self.bands.items():
                idx = (freqs >= fmin) & (freqs <= fmax)
                if np.any(idx):
                    power = float(np.trapz(psd[idx], freqs[idx]))
                else:
                    power = 0.0
                band_powers_all[band_name].append(power)

            spectral_entropies.append(self.compute_spectral_entropy(psd))

        feats = []
        # band powers (per-channel)
        for band_name in self.bands:
            feats.extend([float(v) for v in band_powers_all[band_name]])

        # spectral entropy (per-channel)
        feats.extend([float(v) for v in spectral_entropies])

        # band ratios (global mean powers)
        mean_band_powers = {b: float(np.mean(v)) for b, v in band_powers_all.items()}
        feats.extend(self.compute_band_ratios(mean_band_powers))

        # connectivity (global + first few channels)
        feats.extend(self.compute_connectivity(x))

        # temporal dynamics
        feats.extend(self.compute_temporal_dynamics(x))

        return np.asarray(feats, dtype=np.float64)

    @staticmethod
    def create_nonlinear_gt(times, seizure_time, transition_point_sec=1200, sharpness=3.0):
        """
        Sigmoid GT:
        time_to_seizure = seizure_time - times
        gt = 1 / (1 + exp(k * (t - t0) / t0))
        """
        times = np.asarray(times, dtype=np.float64)
        time_to_seizure = seizure_time - times
        t0 = float(transition_point_sec)
        k = float(sharpness)
        gt = 1.0 / (1.0 + np.exp(k * (time_to_seizure - t0) / (t0 + 1e-12)))
        return gt

    @staticmethod
    def create_piecewise_gt(times, seizure_time, seed=123):
        """
        Piecewise GT (clinical staging):
        - interictal (>30min): ~0.1 +/- noise
        - early preictal (15-30min): 0.2 -> 0.5
        - late preictal (<15min): 0.5 -> 1.0
        """
        rng = np.random.RandomState(seed)
        times = np.asarray(times, dtype=np.float64)
        time_to_seizure = seizure_time - times
        gt = np.zeros_like(times, dtype=np.float64)

        mask_inter = time_to_seizure > 1800
        if np.any(mask_inter):
            gt[mask_inter] = 0.1 + 0.1 * rng.rand(int(np.sum(mask_inter)))

        mask_early = (time_to_seizure <= 1800) & (time_to_seizure > 900)
        if np.any(mask_early):
            prog = (1800.0 - time_to_seizure[mask_early]) / 900.0
            gt[mask_early] = 0.2 + 0.3 * prog

        mask_late = time_to_seizure <= 900
        if np.any(mask_late):
            prog = (900.0 - time_to_seizure[mask_late]) / 900.0
            gt[mask_late] = 0.5 + 0.5 * prog

        return gt

    def process_file_improved(
        self,
        filename,
        seizure_start_sec,
        total_duration_sec=3600,
        window_sec=2,
        step_sec=1,
        gt_type="sigmoid",
        use_pca=True,
        pca_var_keep=0.95,
        sigmoid_transition_point_sec=1200,
        sigmoid_sharpness=3.0,
    ):
        filepath = os.path.join(self.data_dir, filename)
        print("Reading:", filepath)

        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

        existing_channels = [ch for ch in self.standard_channels if ch in raw.ch_names]
        raw.pick(existing_channels)
        n_channels = len(existing_channels)
        print("Using channels:", n_channels)

        sfreq = float(raw.info["sfreq"])

        start_time = max(0.0, float(seizure_start_sec) - float(total_duration_sec))
        end_time = float(seizure_start_sec)
        dur = end_time - start_time
        print("Time range: %.1fs to %.1fs (duration: %.1fs)" % (start_time, end_time, dur))

        # NOTE: MNE uses inclusive tmax in some contexts; this is fine for feature extraction.
        data = raw.get_data(tmin=start_time, tmax=end_time)

        window_samples = int(round(window_sec * sfreq))
        step_samples = int(round(step_sec * sfreq))
        if window_samples <= 1 or step_samples <= 0:
            raise ValueError("Invalid window/step settings.")

        n_windows = (data.shape[1] - window_samples) // step_samples + 1
        if n_windows <= 0:
            raise ValueError("Not enough samples for the given window/step.")

        print("Extracting features from %d windows..." % n_windows)

        all_features = []
        window_times = []

        for i in tqdm(range(n_windows)):
            s = i * step_samples
            e = s + window_samples
            w = data[:, s:e]

            center_time = start_time + ((s + e) / 2.0) / sfreq
            window_times.append(center_time)

            feats = self.extract_rich_features(w, sfreq)
            all_features.append(feats)

        features_array = np.asarray(all_features, dtype=np.float64)
        times_array = np.asarray(window_times, dtype=np.float64)

        print("Raw feature shape:", features_array.shape)

        # sanitize
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        # normalize
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)

        # optional PCA
        features_out = features_normalized
        pca = None
        if use_pca:
            pca = PCA(n_components=float(pca_var_keep), svd_solver="full")
            features_out = pca.fit_transform(features_normalized)
            print("PCA reduced: %d -> %d features" % (features_array.shape[1], features_out.shape[1]))

        # GT
        if gt_type == "linear":
            time_to_seizure = float(seizure_start_sec) - times_array
            gt = 1.0 - (time_to_seizure / float(total_duration_sec))
            gt = np.clip(gt, 0.0, 1.0)
        elif gt_type == "sigmoid":
            gt = self.create_nonlinear_gt(
                times_array,
                float(seizure_start_sec),
                transition_point_sec=sigmoid_transition_point_sec,
                sharpness=sigmoid_sharpness,
            )
        elif gt_type == "piecewise":
            gt = self.create_piecewise_gt(times_array, float(seizure_start_sec))
        else:
            raise ValueError("Unknown gt_type: %s" % str(gt_type))

        gt = gt.astype(np.float32)
        times_save = times_array.astype(np.float32)

        # Save
        np.save(os.path.join(self.output_dir, "chb01_03_features_rich.npy"), features_normalized.astype(np.float32))
        if use_pca:
            np.save(os.path.join(self.output_dir, "chb01_03_features_pca.npy"), features_out.astype(np.float32))
        np.save(os.path.join(self.output_dir, "chb01_03_gt_%s.npy" % gt_type), gt)
        np.save(os.path.join(self.output_dir, "chb01_03_times.npy"), times_save)

        # also save linear GT for comparison
        time_to_seizure = float(seizure_start_sec) - times_array
        gt_linear = 1.0 - (time_to_seizure / float(total_duration_sec))
        gt_linear = np.clip(gt_linear, 0.0, 1.0).astype(np.float32)
        np.save(os.path.join(self.output_dir, "chb01_03_gt_linear.npy"), gt_linear)

        print("\n--- Processing Complete ---")
        print("Features (rich):", features_normalized.shape)
        if use_pca:
            print("Features (PCA): ", features_out.shape)
        print("GT shape:", gt.shape)
        print("GT range: [%.3f, %.3f]" % (float(gt.min()), float(gt.max())))
        print("Saved to:", self.output_dir)

        # plot GT comparison
        self.plot_gt_comparison(times_array, gt, gt_linear, float(seizure_start_sec))

        return features_out.astype(np.float32), gt, times_save, pca

    def plot_gt_comparison(self, times, gt_main, gt_linear, seizure_time):
        """Plot GT comparison and save figure."""
        import matplotlib.pyplot as plt

        times = np.asarray(times, dtype=np.float64)
        seizure_time = float(seizure_time)

        time_to_seizure_min = (seizure_time - times) / 60.0

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax = axes[0]
        ax.plot(time_to_seizure_min, gt_linear, "b--", linewidth=2, label="Linear GT (baseline)")
        ax.plot(time_to_seizure_min, gt_main, "r-", linewidth=2, label="GT (%s)" % GT_TYPE)
        ax.axvline(SIGMOID_TRANSITION_POINT_SEC / 60.0, linestyle=":", alpha=0.7, label="Transition point")
        ax.set_xlabel("Time to Seizure (minutes)")
        ax.set_ylabel("Ground Truth")
        ax.set_title("A. Ground Truth Comparison")
        ax.legend()
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.hist(gt_linear, bins=50, alpha=0.5, label="Linear GT")
        ax.hist(gt_main, bins=50, alpha=0.5, label="GT (%s)" % GT_TYPE)
        ax.set_xlabel("GT Value")
        ax.set_ylabel("Count")
        ax.set_title("B. GT Distribution")
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "gt_comparison.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("Saved GT comparison plot:", out_path)


def analyze_collapse_cause(features_path, gt_path):
    """Analyze possible causes of trajectory collapse."""
    print("\n" + "=" * 60)
    print("Feature and target diagnostics")
    print("=" * 60)

    features = np.load(features_path)
    gt = np.load(gt_path)

    print("\nFeature shape:", features.shape)
    print("GT shape:", gt.shape)

    # 1) GT linearity
    x = np.arange(len(gt), dtype=np.float64)
    gt_f = gt.astype(np.float64)
    corr = np.corrcoef(x, gt_f)[0, 1]
    print("\n1) GT Linearity: r = %.4f" % float(corr))
    if corr > 0.95:
        print("   [WARN] GT is highly linear -> can drive 1D collapse.")

    # 2) feature correlation
    feat = features.astype(np.float64)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    corr_matrix = np.corrcoef(feat.T)
    iu = np.triu_indices_from(corr_matrix, k=1)
    mean_corr = float(np.mean(np.abs(corr_matrix[iu])))
    print("\n2) Mean |Feature-Feature| Correlation: %.4f" % mean_corr)
    if mean_corr > 0.5:
        print("   [WARN] Features are highly correlated -> consider decorrelation (PCA/whitening).")

    # 3) feature-GT correlation
    feat_gt_corrs = []
    for i in range(feat.shape[1]):
        r = np.corrcoef(feat[:, i], gt_f)[0, 1]
        if not np.isnan(r):
            feat_gt_corrs.append(abs(float(r)))
    mean_feat_gt = float(np.mean(feat_gt_corrs)) if len(feat_gt_corrs) > 0 else 0.0
    print("\n3) Mean |Feature-GT| Correlation: %.4f" % mean_feat_gt)
    if mean_feat_gt > 0.7:
        print("   [WARN] Features strongly correlate with GT -> trajectory tends to be linear.")

    # 4) monotonicity check
    monotonic_count = 0
    for i in range(feat.shape[1]):
        d = np.diff(feat[:, i])
        if len(d) == 0:
            continue
        if (np.sum(d > 0) > 0.9 * len(d)) or (np.sum(d < 0) > 0.9 * len(d)):
            monotonic_count += 1
    print("\n4) Monotonic Features: %d / %d" % (monotonic_count, feat.shape[1]))
    if monotonic_count > 0.5 * feat.shape[1]:
        print("   [WARN] Many features are monotonic -> lacks nonlinear dynamics.")

    print("\n" + "=" * 60)
    print("Diagnostic notes:")
    print("=" * 60)
    print("1) Use sigmoid/piecewise GT instead of linear")
    print("2) Add nonlinear features (entropy, connectivity, temporal dynamics)")
    print("3) Apply PCA decorrelation (or whitening) if correlations are high")
    print("4) Ensure training has enough baseline-like segments (GT near 0)")
    print("5) If still too smooth/linear, reduce temporal smoothness weight in training")


# ============================================================================
# Bio-PD EEG analysis pipeline
# ============================================================================

# ============================================================================
# PART 0: CONSTANTS & PALETTE
# ============================================================================
STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
]

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0), "gamma": (30.0, 40.0),
}
BAND_KEYS = list(BANDS.keys())

HEALTHY    = "#2166AC"
TRANSITION = "#FFA500"
DISEASE    = "#B2182B"
PALETTE = {
    "baseline": HEALTHY, "transition": TRANSITION, "ictal": DISEASE,
    "biomarker": "#1A5276", "threshold": "#7B241C", "lead_time": "#27AE60",
    "roc_fill": "#AED6F1", "confidence": "#D5D8DC", "accent": "#E74C3C",
    "distance": "#8E44AD",
}
CMAP_STATE = ListedColormap([HEALTHY, TRANSITION, DISEASE])


# ============================================================================
# PART 1: CHB-MIT SUMMARY PARSER
# ============================================================================
def parse_chb_summary(summary_path):
    """
    Parse chbXX-summary.txt to extract seizure info for all files.
    Returns dict: {filename: [(start_sec, end_sec), ...] or None}
    """
    seizure_info = {}
    if not os.path.exists(summary_path):
        return seizure_info

    with open(summary_path, "r", errors="replace") as f:
        text = f.read()

    # Split into file blocks
    blocks = re.split(r"(?=File Name:)", text)
    for block in blocks:
        fname_match = re.search(r"File Name:\s*(\S+)", block)
        if not fname_match:
            continue
        fname = fname_match.group(1).strip()

        n_seiz_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        n_seiz = int(n_seiz_match.group(1)) if n_seiz_match else 0

        if n_seiz == 0:
            seizure_info[fname] = None
        else:
            seizures = []
            # Match patterns like "Seizure Start Time: 2996 seconds" or "Seizure 1 Start Time: ..."
            starts = re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds", block)
            ends   = re.findall(r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds", block)
            for s, e in zip(starts, ends):
                seizures.append((int(s), int(e)))
            seizure_info[fname] = seizures if seizures else None

    return seizure_info


def discover_all_subjects(data_root):
    """
    Discover all CHB-MIT subjects under data_root.
    Returns list of dicts with subject info.
    """
    subjects = []
    for subj_dir in sorted(glob.glob(os.path.join(data_root, "chb*"))):
        if not os.path.isdir(subj_dir):
            continue
        subj_id = os.path.basename(subj_dir)

        # Find summary file
        summary_candidates = glob.glob(os.path.join(subj_dir, "*summary*"))
        if not summary_candidates:
            print("  [SKIP] %s: no summary file" % subj_id)
            continue
        summary_path = summary_candidates[0]

        # Parse seizure info
        seizure_info = parse_chb_summary(summary_path)
        if not seizure_info:
            print("  [SKIP] %s: empty seizure info" % subj_id)
            continue

        # Find EDF files
        edf_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(subj_dir, "*.edf"))])
        if not edf_files:
            print("  [SKIP] %s: no EDF files" % subj_id)
            continue

        n_seizure_files = sum(1 for f in edf_files if seizure_info.get(f) is not None)
        subjects.append({
            "id": subj_id,
            "dir": subj_dir,
            "summary": summary_path,
            "seizure_info": seizure_info,
            "edf_files": edf_files,
            "n_files": len(edf_files),
            "n_seizure_files": n_seizure_files,
        })
        print("  Found %s: %d files (%d with seizures)" % (subj_id, len(edf_files), n_seizure_files))

    return subjects


# ============================================================================
# PART 2: PREPROCESSING (reused from original, per-subject)
# ============================================================================
PREPROCESS_CONFIG = {
    "WINDOW_SEC": 1.0, "STEP_SEC": 1.0,
    "TRANSITION_SEC": 5 * 60, "BASELINE_GAP_SEC": 30,
    "RISK_SHAPE": "linear", "SIGMOID_SHARPNESS": 6.0,
    "APPLY_BANDPASS": True, "BP_LO": 0.5, "BP_HI": 40.0,
    "APPLY_NOTCH": True, "NOTCH_FREQS": (60.0,),
    "TARGET_CHANNEL_COUNT": 21,
}


def _batch_window_data(data, win_samp, step_samp):
    n_channels, n_samples = data.shape
    n_windows = (n_samples - win_samp) // step_samp + 1
    if n_windows <= 0:
        return None, 0
    from numpy.lib.stride_tricks import as_strided
    byte = data.strides[1]
    ch_str = data.strides[0]
    shape = (n_windows, n_channels, win_samp)
    strides = (step_samp * byte, ch_str, byte)
    return as_strided(data, shape=shape, strides=strides), n_windows


def _extract_connectivity_batch(windows, topk_list=(4, 8, 16, 32)):
    n_windows, n_channels, win_samp = windows.shape
    if n_channels < 2:
        return np.zeros((n_windows, 4 + len(topk_list)), dtype=np.float32)
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + 1e-12
    normed = (windows - mean) / std
    corr = np.einsum("wct,wdt->wcd", normed, normed) / win_samp
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.arange(n_channels)
    corr[:, diag, diag] = 0.0
    r, c = np.triu_indices(n_channels, k=1)
    vals = np.abs(corr[:, r, c])
    n_pairs = vals.shape[1]
    if n_pairs == 0:
        return np.zeros((n_windows, 4 + len(topk_list)), dtype=np.float32)
    out = [vals.mean(axis=1, keepdims=True), np.median(vals, axis=1, keepdims=True),
           vals.max(axis=1, keepdims=True), vals.std(axis=1, keepdims=True)]
    vals_sorted = np.sort(vals, axis=1)[:, ::-1]
    for k in topk_list:
        kk = min(k, n_pairs)
        out.append(vals_sorted[:, :kk].mean(axis=1, keepdims=True))
    return np.concatenate(out, axis=1).astype(np.float32)


def _extract_features_vectorized(windows, sfreq, nperseg=256):
    n_windows, n_channels, win_samp = windows.shape
    nperseg = min(nperseg, win_samp)
    all_feats = []
    for ch in range(n_channels):
        ch_data = windows[:, ch, :]
        ll = np.sum(np.abs(np.diff(ch_data, axis=1)), axis=1)
        all_feats.append(ll[:, np.newaxis])
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=0, axis=-1)
        psd = np.maximum(psd, 1e-18)
        total = np.trapz(psd, freqs, axis=-1) + 1e-12
        band_cols = []
        for bk in BAND_KEYS:
            fmin, fmax = BANDS[bk]
            idx = (freqs >= fmin) & (freqs <= fmax)
            bp = np.trapz(psd[:, idx], freqs[idx], axis=-1) if np.any(idx) else np.zeros(n_windows)
            band_cols.append(bp / total)
        all_feats.append(np.column_stack(band_cols))
    conn_feats = _extract_connectivity_batch(windows)
    all_feats.append(conn_feats)
    return np.concatenate(all_feats, axis=1).astype(np.float32)


def make_gt_three_state_and_risk(center_times, seizure_times_list, transition_sec=300, baseline_gap_sec=30):
    """
    Generate GT state and risk for a file.
    seizure_times_list: list of (start, end) tuples, or None
    """
    t = np.asarray(center_times, dtype=np.float64)
    gt_state = np.zeros_like(t, dtype=np.int64)
    gt_risk = np.zeros_like(t, dtype=np.float32)

    if seizure_times_list is None:
        return gt_state, gt_risk, {"has_seizure": False}

    for s_start, s_end in seizure_times_list:
        s_start, s_end = float(s_start), float(s_end)
        trans_e = s_start
        trans_s = max(0.0, s_start - float(transition_sec))

        m_ictal = (t >= s_start) & (t <= s_end)
        gt_state[m_ictal] = 2
        gt_risk[m_ictal] = 1.0

        m_trans = (t >= trans_s) & (t < trans_e) & (gt_state == 0)
        gt_state[m_trans] = 1
        z = np.clip((t[m_trans] - trans_s) / max(1e-9, trans_e - trans_s), 0, 1)
        gt_risk[m_trans] = np.maximum(gt_risk[m_trans], z.astype(np.float32))

    has_seizure = np.any(gt_state == 2)
    return gt_state, gt_risk.astype(np.float32), {"has_seizure": bool(has_seizure)}


def preprocess_subject(subj_info, cfg=None):
    """
    Preprocess a single subject: load EDFs, extract features, generate GT.
    Returns dict with features, gt_state, gt_risk, seg_infos, etc.
    """
    if cfg is None:
        cfg = PREPROCESS_CONFIG
    target_n_ch = int(cfg["TARGET_CHANNEL_COUNT"])
    subj_id = subj_info["id"]
    data_dir = subj_info["dir"]
    seizure_info = subj_info["seizure_info"]

    print("\n[PREPROCESS] %s (%d files)" % (subj_id, len(subj_info["edf_files"])))

    # Pass 1: find common channels
    channel_sets = []
    valid_files = []
    for edf_file in subj_info["edf_files"]:
        edf_path = os.path.join(data_dir, edf_file)
        if not os.path.exists(edf_path):
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            existing = [ch for ch in STANDARD_CHANNELS if ch in raw.ch_names]
            if len(existing) >= target_n_ch:
                channel_sets.append(set(existing))
                valid_files.append(edf_file)
            del raw
        except Exception as e:
            print("  SKIP %s: %s" % (edf_file, str(e)))

    if len(valid_files) == 0 or len(channel_sets) == 0:
        print("  [SKIP] %s: no valid files" % subj_id)
        return None

    common_channels = list(sorted(
        set.intersection(*channel_sets),
        key=lambda ch: STANDARD_CHANNELS.index(ch) if ch in STANDARD_CHANNELS else 999,
    ))
    chosen = common_channels[:target_n_ch]
    n_channels = len(chosen)
    if n_channels < target_n_ch:
        print("  [SKIP] %s: only %d common channels" % (subj_id, n_channels))
        return None

    # Pass 2: load and process per-file (NOT concatenated across files to save memory)
    # We process each file independently and store features + GT
    win_sec = float(cfg["WINDOW_SEC"])
    step_sec = float(cfg["STEP_SEC"])

    all_features = []
    all_gt_state = []
    all_gt_risk = []
    seg_infos = []
    sfreq = None
    global_offset = 0

    for fi, edf_file in enumerate(valid_files):
        edf_path = os.path.join(data_dir, edf_file)
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.pick(chosen)
            if sfreq is None:
                sfreq = float(raw.info["sfreq"])

            # Filter
            if bool(cfg.get("APPLY_NOTCH", True)):
                raw.notch_filter(list(cfg.get("NOTCH_FREQS", (60.0,))), verbose=False)
            if bool(cfg.get("APPLY_BANDPASS", True)):
                raw.filter(float(cfg.get("BP_LO", 0.5)), float(cfg.get("BP_HI", 40.0)), verbose=False)

            data = raw.get_data()
            del raw

            win_samp = int(round(win_sec * sfreq))
            step_samp = int(round(step_sec * sfreq))
            windows, n_windows = _batch_window_data(data, win_samp, step_samp)
            if n_windows <= 0:
                del data
                continue

            # Center times (in seconds within this file)
            starts = np.arange(n_windows) * step_samp
            ends = starts + win_samp
            centers = ((starts + ends) / 2.0) / sfreq

            # Extract features
            nperseg = min(256, win_samp)
            CHUNK = 2000
            feat_chunks = []
            for c0 in range(0, n_windows, CHUNK):
                c1 = min(c0 + CHUNK, n_windows)
                chunk = np.ascontiguousarray(windows[c0:c1])
                feat_chunks.append(_extract_features_vectorized(chunk, sfreq, nperseg))
            X = np.concatenate(feat_chunks, axis=0)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            del data, windows, feat_chunks
            gc.collect()

            # GT labels
            file_seizures = seizure_info.get(edf_file, None)
            gt_s, gt_r, meta = make_gt_three_state_and_risk(
                centers, file_seizures,
                transition_sec=int(cfg["TRANSITION_SEC"]),
                baseline_gap_sec=int(cfg["BASELINE_GAP_SEC"]),
            )

            seg_infos.append({
                "file": edf_file, "subject": subj_id,
                "has_seizure": meta["has_seizure"],
                "start": global_offset, "end": global_offset + n_windows,
                "gt_state": gt_s, "gt_risk": gt_r, "n_windows": n_windows,
            })

            all_features.append(X)
            all_gt_state.append(gt_s)
            all_gt_risk.append(gt_r)
            global_offset += n_windows

        except Exception as e:
            print("  ERROR %s: %s" % (edf_file, str(e)))
            continue

    if len(all_features) == 0:
        return None

    features = np.concatenate(all_features, axis=0)
    gt_state = np.concatenate(all_gt_state, axis=0)
    gt_risk = np.concatenate(all_gt_risk, axis=0)

    # Standardise features (per-subject)
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)

    n_bl = int(np.sum(gt_state == 0))
    n_tr = int(np.sum(gt_state == 1))
    n_ic = int(np.sum(gt_state == 2))
    n_seiz_files = sum(1 for s in seg_infos if s["has_seizure"])
    print("  %s: T=%d, F=%d, files=%d (seizure=%d) | BL=%d TR=%d IC=%d"
          % (subj_id, features.shape[0], features.shape[1],
             len(seg_infos), n_seiz_files, n_bl, n_tr, n_ic))

    return {
        "subject": subj_id,
        "features": features,
        "gt_state": gt_state,
        "gt_risk": gt_risk,
        "seg_infos": seg_infos,
        "sfreq": float(sfreq),
        "n_channels": n_channels,
        "scaler": scaler,
    }


# ============================================================================
# PART 3: CROSS-SUBJECT ChebGCN PRETRAINING (Self-Supervised)
# ============================================================================
# The encoder learns to map EEG features -> graph-refined embeddings
# using reconstruction loss (no labels needed).
# Each subject contributes mini-batches of windowed graph data.

def _reorganize_features_by_channel(features, n_channels, n_bands=9999):
    T, S = features.shape
    if S >= n_channels * n_bands:
        ch_feat = features[:, :n_channels * n_bands].reshape(T, n_channels, n_bands)
    else:
        fpc = max(1, S // n_channels)
        ch_feat = np.zeros((T, n_channels, fpc), dtype=np.float32)
        for ch in range(n_channels):
            s = ch * fpc
            e = min(s + fpc, S)
            if s < S:
                ch_feat[:, ch, :(e - s)] = features[:, s:e]
    return ch_feat.astype(np.float32)


def _build_edges_vectorized(T, n_channels, top_k, channel_features):
    """Build temporal + spatial edges."""
    t_idx = np.arange(T - 1)
    ch_idx = np.arange(n_channels)
    t_grid, ch_grid = np.meshgrid(t_idx, ch_idx, indexing='ij')
    n1 = (t_grid * n_channels + ch_grid).ravel()
    n2 = ((t_grid + 1) * n_channels + ch_grid).ravel()
    src_temporal = np.concatenate([n1, n2])
    dst_temporal = np.concatenate([n2, n1])

    ch_ts = channel_features.mean(axis=2)
    corr = np.corrcoef(ch_ts, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    np.fill_diagonal(corr, 0.0)

    spatial_pairs = []
    for ch in range(n_channels):
        nbs = np.argsort(-corr[ch])[:top_k]
        for nb in nbs:
            if corr[ch, int(nb)] >= 0.1:
                spatial_pairs.append((ch, int(nb)))

    if spatial_pairs:
        sp = np.array(spatial_pairs, dtype=np.int64)
        all_t = np.arange(T, dtype=np.int64)[:, None]
        src_sp = (all_t * n_channels + sp[:, 0][None, :]).ravel()
        dst_sp = (all_t * n_channels + sp[:, 1][None, :]).ravel()
        src_all = np.concatenate([src_temporal, src_sp, dst_sp])
        dst_all = np.concatenate([dst_temporal, dst_sp, src_sp])
    else:
        src_all, dst_all = src_temporal, dst_temporal

    edge_pairs = np.stack([src_all, dst_all], axis=0)
    max_node = T * n_channels
    codes = edge_pairs[0].astype(np.int64) * max_node + edge_pairs[1].astype(np.int64)
    _, unique_idx = np.unique(codes, return_index=True)
    return edge_pairs[:, unique_idx].astype(np.int64)


class SharedChebGCNEncoder(nn.Module):
    """
    Shared ChebGCN encoder that can be pretrained across subjects.
    Architecture: 2-layer ChebConv encoder + linear decoder (for pretraining).
    """
    def __init__(self, in_channels, hidden_dim=16, out_channels=8, K=3):
        super().__init__()
        self.cheb1 = ChebConv(in_channels, hidden_dim, K=K)
        self.cheb2 = ChebConv(hidden_dim, out_channels, K=K)
        self.decoder = nn.Linear(out_channels, in_channels)

    def encode(self, x, edge_index, edge_weight=None):
        h = F.relu(self.cheb1(x, edge_index, edge_weight))
        h = self.cheb2(h, edge_index, edge_weight)
        return h

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.encode(x, edge_index, edge_weight)
        recon = self.decode(h)
        return recon, h


def _prepare_graph_chunk(features_chunk, n_channels, top_k=5, n_bands=9999):
    """
    Prepare a graph from a chunk of features.
    Returns: (x_tensor, edge_index_tensor, T, in_ch)
    """
    T, S = features_chunk.shape
    ch_feat = _reorganize_features_by_channel(features_chunk, n_channels, n_bands)
    in_ch = ch_feat.shape[2]
    edge_index = _build_edges_vectorized(T, n_channels, top_k, ch_feat)
    x = ch_feat.reshape(T * n_channels, in_ch).astype(np.float32)
    return (torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(edge_index, dtype=torch.long),
            T, in_ch)


def pretrain_cross_subject_chebgcn(all_subject_data, config, device, save_path):
    """
    Phase A: Pretrain ChebGCN encoder across ALL subjects using self-supervised
    reconstruction loss. No labels used.

    Strategy:
      - Sample random chunks (size ~500 windows) from each subject
      - Train encoder to reconstruct node features from graph embeddings
      - Iterate across subjects for multiple epochs
    """
    n_channels = int(config.n_channels)
    chunk_size = int(config.pretrain_chunk_size)
    n_epochs = int(config.pretrain_epochs)
    lr = float(config.pretrain_lr)
    top_k = int(config.top_k_correlations)

    # Determine in_channels from first subject
    sample_subj = all_subject_data[0]
    ch_feat = _reorganize_features_by_channel(
        sample_subj["features"][:10], n_channels, 9999
    )
    in_ch = ch_feat.shape[2]
    print("[PRETRAIN] in_channels=%d, hidden=%d, out=%d, K=%d"
          % (in_ch, int(config.cheb_hidden_dim), int(config.cheb_out_channels), int(config.cheb_K)))

    model = SharedChebGCNEncoder(
        in_channels=in_ch,
        hidden_dim=int(config.cheb_hidden_dim),
        out_channels=int(config.cheb_out_channels),
        K=int(config.cheb_K),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    loss_fn = nn.MSELoss()

    n_subjects = len(all_subject_data)
    print("[PRETRAIN] %d subjects, %d epochs, chunk_size=%d"
          % (n_subjects, n_epochs, chunk_size))

    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        # Shuffle subject order each epoch
        subj_order = list(range(n_subjects))
        random.shuffle(subj_order)

        for si in subj_order:
            subj = all_subject_data[si]
            features = subj["features"]
            T_total = features.shape[0]
            if T_total < 50:
                continue

            # Sample random chunks from this subject
            n_chunks = max(1, T_total // chunk_size)
            for _ in range(min(n_chunks, 5)):  # max 5 chunks per subject per epoch
                start = random.randint(0, max(0, T_total - chunk_size))
                end = min(start + chunk_size, T_total)
                if end - start < 30:
                    continue

                chunk_data = features[start:end]
                x_t, ei, _T, _ic = _prepare_graph_chunk(chunk_data, n_channels, top_k)
                x_t = x_t.to(device)
                ei = ei.to(device)

                optimizer.zero_grad()
                recon, _emb = model(x_t, ei)
                loss = loss_fn(recon, x_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1

                del x_t, ei, recon, _emb
                gc.collect()

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_steps)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("  Epoch %d/%d | Loss: %.6f (best: %.6f) | steps: %d"
                  % (epoch + 1, n_epochs, avg_loss, best_loss, n_steps))

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Save pretrained encoder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("[PRETRAIN] Saved to %s (best_loss=%.6f)" % (save_path, best_loss))

    return model, in_ch


def _apply_timecorr_smoothing(data_chunk, smooth_window=1, max_radius=30):
    """
    Apply BOUNDED timecorr-based temporal smoothing to a data chunk.

    The key insight: full timecorr (the earlier implementation) averages across hundreds of seconds,
    which creates beautiful continuous trajectories but DESTROYS state boundaries
    (transition/ictal signals get smeared into baseline). We need a middle ground:

    Strategy: ACF-weighted LOCAL smoothing with a hard radius cap.
    - Compute ACF to get data-driven smoothing weights
    - But cap the smoothing radius to max_radius seconds
    - This preserves LOCAL continuity (points flow smoothly)
    - While maintaining SHARP transitions at state boundaries

    Args:
        data_chunk: (T, S) feature array
        smooth_window: ACF smoothing kernel size
        max_radius: maximum smoothing radius in time steps (default 30 = 30 seconds)
    """
    T, S = data_chunk.shape
    if T <= 10:
        return data_chunk

    # Compute per-feature ACF (only up to max_radius lags, not T-1)
    max_lag = min(max_radius, T - 1)
    A_feat = np.empty((max_lag + 1, S), dtype=np.float64)
    for f in range(S):
        A_feat[:, f] = sm.tsa.acf(data_chunk[:, f], fft=False, nlags=max_lag, missing="drop")
    A_mean = np.nanmean(A_feat, axis=1)  # (max_lag+1,) ACF values

    if smooth_window > 1:
        A_mean = np.convolve(A_mean, np.ones(int(smooth_window)), "same") / float(smooth_window)

    # Find effective radius: where ACF drops below 0 (or max_radius)
    drop_idx = np.where(A_mean[1:] < 0)[0]  # skip lag-0 (always 1.0)
    effective_radius = int(drop_idx[0]) if len(drop_idx) > 0 else max_lag
    effective_radius = min(effective_radius, max_radius)

    if effective_radius < 1:
        return data_chunk

    # Build BANDED M-matrix: only +/-effective_radius neighbours
    # Use sparse-style row-by-row construction to avoid TxT memory for large T
    smoothed = np.empty_like(data_chunk)
    acf_weights = np.maximum(A_mean[:effective_radius + 1], 0.0)  # only positive ACF
    acf_weights[0] = 0.0  # exclude self (will be added separately)

    for t in range(T):
        lo = max(0, t - effective_radius)
        hi = min(T, t + effective_radius + 1)

        # Compute weights for neighbours
        lags = np.abs(np.arange(lo, hi) - t)
        w = acf_weights[lags]

        # Normalise: self-weight = 1, neighbour weights sum to blend_factor
        w_sum = w.sum()
        if w_sum > 1e-12:
            # Self-retention ratio: higher = less smoothing, more sharp boundaries
            # With max_radius=30 and typical ACF, this gives ~60-70% self-retention
            self_weight = 1.0
            w = w / w_sum  # normalise neighbour weights to sum to 1
            blend = min(w_sum / (w_sum + self_weight), 0.5)  # cap blend at 50%
            smoothed[t] = (1.0 - blend) * data_chunk[t] + blend * (w @ data_chunk[lo:hi])
        else:
            smoothed[t] = data_chunk[t]

    return smoothed.astype(np.float32)


def refine_with_pretrained_chebgcn(features, pretrained_model, config, device):
    """
    Use pretrained ChebGCN encoder to refine features for a single subject.
    Chunked processing with timecorr smoothing for trajectory continuity.
    """
    T, S = features.shape
    n_ch = int(config.n_channels)
    out_ch = int(config.cheb_out_channels)
    chunk_sz = int(config.cheb_chunk_size)
    overlap = int(config.cheb_chunk_overlap)
    top_k = int(config.top_k_correlations)
    use_timecorr = bool(getattr(config, 'use_timecorr_in_chunks', True))
    smooth_window = int(getattr(config, 'smooth_window', 1))
    max_radius = int(getattr(config, 'timecorr_max_radius', 30))

    pretrained_model.eval()
    refined_full = np.zeros((T, n_ch, out_ch), dtype=np.float32)
    weight_full = np.zeros((T, 1, 1), dtype=np.float32)

    step = chunk_sz - overlap
    n_chunks = max(1, (T - overlap + step - 1) // step)
    print("[ChebGCN-Pretrained] T=%d -> %d chunks (timecorr=%s, radius=%d)"
          % (T, n_chunks, "ON" if use_timecorr else "OFF", max_radius if use_timecorr else 0))

    for ci, c_start in enumerate(range(0, T, step)):
        c_end = min(c_start + chunk_sz, T)
        if c_end - c_start < 30:
            break

        chunk_data = features[c_start:c_end].copy()

        # Bounded timecorr smoothing: local continuity WITHOUT destroying state boundaries
        if use_timecorr and chunk_data.shape[0] > 10:
            chunk_data = _apply_timecorr_smoothing(chunk_data, smooth_window, max_radius)

        x_t, ei, chunk_T, _ic = _prepare_graph_chunk(chunk_data, n_ch, top_k)
        x_t = x_t.to(device)
        ei = ei.to(device)

        with torch.no_grad():
            emb = pretrained_model.encode(x_t, ei)
            refined_chunk = emb.cpu().numpy().reshape(chunk_T, n_ch, out_ch)

        L = c_end - c_start
        w = np.ones(L, dtype=np.float32)
        blend = min(overlap, L // 2)
        if blend > 1 and c_start > 0:
            w[:blend] = np.linspace(0, 1, blend, dtype=np.float32)
        if blend > 1 and c_end < T:
            w[-blend:] = np.linspace(1, 0, blend, dtype=np.float32)

        w3 = w[:, None, None]
        refined_full[c_start:c_end] += refined_chunk * w3
        weight_full[c_start:c_end] += w3

        del x_t, ei, emb
        gc.collect()

        if (ci + 1) % 20 == 0:
            print("  chunk %d/%d" % (ci + 1, n_chunks))

    mask = weight_full[:, 0, 0] > 1e-12
    refined_full[mask] /= weight_full[mask]
    print("[ChebGCN-Pretrained] Done -> (%d, %d, %d)" % (T, n_ch, out_ch))
    return refined_full.astype(np.float32)


# ============================================================================
# PART 4: UNSUPERVISED TRAJECTORY GENERATION
# ============================================================================
# CRITICAL: no state labels are used here. Only KL divergence + temporal smoothness.

class ConfigCrossSubject:
    """Configuration for cross-subject pipeline."""
    def __init__(self):
        self.cuda_device = "0"
        self.seed = 42

        # Cross-subject pretraining
        self.pretrain_epochs = 30
        self.pretrain_chunk_size = 500
        self.pretrain_lr = 1e-3

        # Co-training
        self.n_iterations = 2
        self.n_recur = 4
        self.balance_degree = 50
        self.max_batch_size = 4000    # Cap to prevent OOM on large subjects
        self.batch_size = None
        self.max_epochs = 150
        self.patience = 30
        self.learning_rate = 2e-4
        self.max_grad_norm = 2.0
        self.weight_decay = 1e-5
        self.stage_lr_decay = 0.3

        # Temporal loss (ONLY non-supervised losses)
        self.lambda_temporal = 0.1
        self.temporal_max_lag = 2
        self.smooth_window = 1
        self.context = 32
        self.stride = None
        self.shift_enabled = True
        self.shift_step = 30
        self.shift_num_offsets = 5
        self.p_update_frequency = 10
        self.p_refresh_ema = 0.3
        self.lambda_overlap = 0.02
        self.overlap_warmup_epochs = 20

        # NO state contrastive loss for trajectory generation
        # (This is the key difference from the earlier implementation)

        # Collapse detection
        self.max_retries = 5
        self.var_threshold = 0.3
        self.dim_ratio_threshold = 0.05

        # DR network
        self.num_conv_layers = 4
        self.filters_list = [3, 16, 32, 64]
        self.kernel_size = 3
        self.alpha = 0.05
        self.dense_units = (1024, 512, 256, 8)
        self.final_units = 2

        # ChebGCN
        self.cheb_K = 3
        self.cheb_hidden_dim = 16
        self.cheb_out_channels = 8
        self.cheb_lr = 1e-3
        self.cheb_epochs = 30
        self.cheb_chunk_size = 2000
        self.cheb_chunk_overlap = 500
        self.use_timecorr_in_chunks = True   # Critical for trajectory continuity
        self.timecorr_max_radius = 30        # Bounded smoothing radius (seconds)

        # Graph
        self.top_k_correlations = 5
        self.n_channels = 21
        self.n_bands = 9999

        # Manifold
        self.HD_type = "sherlock"
        self.low_dim = 2
        self.perplexity = 40
        self.p_cache_max_T = 200000

        # Speed
        self.use_gpu_p = True
        self.use_amp = True
        self.p_parallel_workers = 4

        # Paths
        self.data_path = ""
        self.output_dir = ""


# --- GPU P computation (unchanged from original) ---
@torch.no_grad()
def x2p_gpu(X_np, perplexity, device, tol=1e-5, max_iter=50):
    n = X_np.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    target_H = np.log(float(perplexity))
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
    sum_sq = (X * X).sum(dim=1)
    D = sum_sq.unsqueeze(1) + sum_sq.unsqueeze(0) - 2.0 * (X @ X.t())
    D = torch.clamp(D, min=0.0)
    D.fill_diagonal_(0.0)
    beta = torch.ones(n, device=device, dtype=torch.float32)
    beta_min = torch.full((n,), 1e-20, device=device, dtype=torch.float32)
    beta_max = torch.full((n,), 1e20, device=device, dtype=torch.float32)
    inf_mask = torch.ones(n, device=device, dtype=torch.bool)
    target_H_t = torch.tensor(target_H, device=device, dtype=torch.float32)
    diag_mask = 1.0 - torch.eye(n, device=device)
    for _ in range(max_iter):
        exp_vals = torch.exp(-beta.unsqueeze(1) * D) * diag_mask
        row_sum = exp_vals.sum(dim=1, keepdim=True).clamp(min=1e-30)
        P_cond = exp_vals / row_sum
        H = -(P_cond * torch.log(P_cond.clamp(min=1e-30))).sum(dim=1)
        H_diff = H - target_H_t
        converged = H_diff.abs() < tol
        if converged.all():
            break
        too_large = (H_diff > 0) & ~converged
        too_small = (H_diff <= 0) & ~converged
        beta_min = torch.where(too_large, beta, beta_min)
        beta = torch.where(too_large & inf_mask, beta * 2.0,
                           torch.where(too_large, (beta + beta_max) / 2.0, beta))
        beta_max = torch.where(too_small, beta, beta_max)
        inf_mask = inf_mask & ~too_small
        beta = torch.where(too_small, (beta + beta_min) / 2.0, beta)
    exp_vals = torch.exp(-beta.unsqueeze(1) * D) * diag_mask
    row_sum = exp_vals.sum(dim=1, keepdim=True).clamp(min=1e-30)
    P_cond = exp_vals / row_sum
    P = (P_cond + P_cond.t()) / (2.0 * n)
    P_np = P.cpu().numpy().astype(np.float64)
    P_np = np.nan_to_num(P_np, nan=0.0, posinf=0.0, neginf=0.0)
    P_np = np.maximum(P_np, 1e-12)
    return P_np


def _compute_single_P(block, HD_type, perplexity, device=None, use_gpu=False):
    can_gpu = use_gpu and device is not None and str(device).startswith("cuda") and HD_type == "sherlock"
    if can_gpu:
        return x2p_gpu(block, perplexity, device)
    else:
        P = x2p(block, perplexity)
        P[np.isnan(P)] = 0
        P = (P + P.T) / 2.0
        P = P / (P.sum() + 1e-8)
        return np.maximum(P, 1e-12)


def _build_P_cache(X_source, batch_size, context, stride, config, device):
    n = X_source.shape[0]
    use_gpu = bool(getattr(config, "use_gpu_p", True))
    can_gpu = use_gpu and str(device).startswith("cuda") and config.HD_type == "sherlock"
    perplexity = int(config.perplexity)
    jobs = {}
    i = 0
    while i < n:
        end = min(i + batch_size + 2 * context, n)
        cs = min(i + context, end)
        Bc = min(batch_size, end - cs)
        if Bc >= 2:
            key = (int(cs), int(Bc))
            if key not in jobs:
                jobs[key] = X_source[cs:cs + Bc].reshape(Bc, -1).copy()
        i += stride
    cache = {}
    if can_gpu:
        for key, feats in jobs.items():
            cache[key] = x2p_gpu(feats, perplexity, device)
    else:
        for key, feats in jobs.items():
            cache[key] = _compute_single_P(feats, config.HD_type, perplexity, device, False)
    return cache


def _p_cache_to_gpu(cache, device):
    """
    Move P cache to GPU. Falls back to CPU if GPU memory is insufficient.
    Returns (gpu_cache_dict, on_gpu_bool).
    """
    if not str(device).startswith("cuda"):
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in cache.items()}, False

    # Estimate total memory needed
    total_bytes = sum(v.nbytes for v in cache.values())
    total_mb = total_bytes / (1024 ** 2)

    # Check available GPU memory
    try:
        free_mem = torch.cuda.mem_get_info(0)[0] / (1024 ** 2)  # free MB
        if total_mb > free_mem * 0.5:  # don't use more than 50% of free mem
            print("  [P-cache] %.0f MB needed, %.0f MB free -> keeping on CPU" % (total_mb, free_mem))
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in cache.items()}, False
    except Exception:
        pass

    try:
        gpu_cache = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in cache.items()}
        return gpu_cache, True
    except torch.cuda.OutOfMemoryError:
        print("  [P-cache] GPU OOM -> falling back to CPU cache")
        torch.cuda.empty_cache()
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in cache.items()}, False


# --- Utility functions ---
def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def check_embedding_collapse(Y, var_threshold=0.3, dim_ratio_threshold=0.05):
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()
    if Y.shape[0] < 2:
        return False, "Too few"
    var = Y.var(dim=0).sum().item()
    if var < var_threshold:
        return True, "Var low (%.4f)" % var
    Yc = Y - Y.mean(dim=0, keepdim=True)
    try:
        _U, S, _V = torch.linalg.svd(Yc, full_matrices=False)
        if len(S) > 1 and S[0] > 1e-6:
            r = (S[1] / S[0]).item()
            if r < dim_ratio_threshold:
                return True, "1D (%.4f)" % r
    except:
        pass
    return False, "OK (%.2f)" % var


def eeg_data_para(data, balance_degree=4, max_batch_size=4000):
    n_all = data.shape[0]
    degree = max(1, int(balance_degree))
    batch_size = n_all // degree
    # Cap batch_size to prevent OOM (P matrix = batch_size2 x 4B)
    if batch_size > max_batch_size:
        batch_size = max_batch_size
        degree = n_all // batch_size
    n = batch_size * degree
    data_truncated = data[:n]
    X = (data_truncated - np.mean(data_truncated, axis=0)) / (np.std(data_truncated, axis=0) + 1e-8)
    return X.astype(np.float32), int(batch_size), int(n)


def create_kl_divergence_stable(low_dim=2):
    def KLdivergence(P, Y):
        alpha = low_dim - 1
        eps = 1e-15
        sY = torch.sum(Y ** 2, dim=1)
        D = sY[:, None] + sY[None, :] - 2 * (Y @ Y.t())
        D = torch.clamp(D, min=0)
        Q = torch.pow(1 + D / alpha, -(alpha + 1) / 2)
        mask = 1 - torch.eye(Y.shape[0], device=Y.device)
        Q = Q * mask
        Qs = torch.sum(Q)
        Q = Q / Qs if Qs > eps else mask / (Y.shape[0] * (Y.shape[0] - 1))
        Q = torch.clamp(Q, min=eps, max=1.0)
        mp = P > eps
        kl = torch.zeros_like(P)
        kl[mp] = P[mp] * torch.log(P[mp] / (Q[mp] + eps))
        return torch.clamp(torch.sum(kl), max=50.0)
    return KLdivergence


def make_acf_weights(X, max_lag=2, smooth_window=1):
    n, f = X.shape
    acfs = []
    for k in range(f):
        acfs.append(sm.tsa.acf(X[:, k], fft=False, nlags=max_lag, missing="drop"))
    w = np.clip(np.nanmean(np.stack(acfs), axis=0)[1:], 0, None)
    if smooth_window > 1:
        w = np.convolve(w, np.ones(smooth_window), "same") / float(smooth_window)
    w = w / (w.sum() + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


def temporal_smoothness_loss(Y, acf_w):
    B = Y.shape[0]
    L = int(acf_w.numel())
    if L == 0 or B < 2:
        return Y.new_tensor(0.0)
    loss = Y.new_tensor(0.0)
    for lag in range(1, L + 1):
        if B - lag <= 0:
            break
        loss = loss + acf_w[lag - 1] * (Y[:-lag] - Y[lag:]).pow(2).mean()
    return loss


def get_epoch_offset(epoch, config):
    if not bool(config.shift_enabled) or config.batch_size is None:
        return 0
    idx = (epoch // max(1, int(config.p_update_frequency))) % max(1, int(config.shift_num_offsets))
    step = min(int(config.shift_step), max(1, int(config.batch_size) - 1))
    return int((idx * step) % int(config.batch_size))


class EarlyStopper:
    def __init__(self, patience=30, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
        self.state = None

    def step(self, value, model):
        if float(value) < self.best - self.min_delta:
            self.best = float(value)
            self.count = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.count += 1
        return self.count >= self.patience


def extract_features_block(model, X_block_t, layer_name=None):
    if layer_name is None:
        return None
    layer_map = {"Dense1": 1, "Dense2": 2, "Dense3": 3}
    layer_num = int(layer_map.get(layer_name, 1))
    with torch.no_grad():
        feats = model.get_layer_output(X_block_t, layer_num)
    return feats.detach().cpu().numpy().reshape(feats.shape[0], -1)


def _extract_all_intermediate_features(model, X_all_perm, n, layer_name, device, chunk_size=256):
    model.eval()
    feat_chunks = []
    with torch.no_grad():
        for i in range(0, n, chunk_size):
            e = min(i + chunk_size, n)
            feat = extract_features_block(model, X_all_perm[i:e], layer_name)
            feat_chunks.append(feat)
    feats = np.concatenate(feat_chunks, axis=0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    feats = np.clip(feats, -1e6, 1e6)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-8
    return ((feats - mu) / sd).astype(np.float32)


def train_unsupervised_dr(model, X_train_auto, out_model_name, config, device,
                          layer_name=None, stage_index=0):
    """
    PURELY UNSUPERVISED DR training.
    Loss = L_kl + lambda_temporal * L_temporal + lambda_overlap * L_overlap
    NO state contrastive loss.
    """
    model.to(device)
    stage_lr = float(config.learning_rate) * (float(config.stage_lr_decay) ** stage_index)
    print("  [LR] Stage %d: %.1e" % (stage_index + 1, stage_lr))

    optimizer = AdamW(model.parameters(), lr=stage_lr, weight_decay=float(config.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config.max_epochs), eta_min=stage_lr * 0.01)
    criterion = create_kl_divergence_stable(int(config.low_dim))

    n = X_train_auto.shape[0]
    batch_size = int(config.batch_size)
    stride = max(1, batch_size // 2) if layer_name is not None else max(1, batch_size // 2)
    context = max(0, int(config.context))

    X_all_t = torch.as_tensor(X_train_auto, dtype=torch.float32, device=device)
    X_all_perm = X_all_t.permute(0, 3, 1, 2).contiguous()

    X_flat = X_train_auto.reshape(n, -1)
    acf_w = make_acf_weights(X_flat, int(config.temporal_max_lag), int(config.smooth_window)).to(device)

    stopper = EarlyStopper(patience=int(config.patience))
    use_amp = bool(config.use_amp) and str(device).startswith("cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Build P cache
    if layer_name is None:
        P_cache = _build_P_cache(X_flat, batch_size, context, stride, config, device)
    else:
        inter_feats = _extract_all_intermediate_features(model, X_all_perm, n, layer_name, device)
        P_cache = _build_P_cache(inter_feats, batch_size, context, stride, config, device)
        del inter_feats
    P_cache_gpu, _p_on_gpu = _p_cache_to_gpu(P_cache, device)
    del P_cache
    print("  [P-cache] %d entries (%s)" % (len(P_cache_gpu), "GPU" if _p_on_gpu else "CPU"))

    p_update_freq = max(1, int(config.p_update_frequency))
    last_p_refresh = 0

    for epoch in range(int(config.max_epochs)):
        # P refresh for later stages
        if layer_name is not None and epoch > 0 and (epoch - last_p_refresh) >= p_update_freq:
            inter_feats = _extract_all_intermediate_features(model, X_all_perm, n, layer_name, device)
            P_fresh = _build_P_cache(inter_feats, batch_size, context, stride, config, device)
            P_fresh_gpu, _ = _p_cache_to_gpu(P_fresh, device)
            del P_fresh, inter_feats
            beta = float(config.p_refresh_ema)
            for key in P_fresh_gpu:
                if key in P_cache_gpu:
                    P_cache_gpu[key] = beta * P_fresh_gpu[key] + (1 - beta) * P_cache_gpu[key]
                else:
                    P_cache_gpu[key] = P_fresh_gpu[key]
            del P_fresh_gpu
            last_p_refresh = epoch
            gc.collect()

        model.train()
        epoch_loss = 0.0
        steps = 0
        prev_start = prev_end = None
        prev_Y = None

        offset = get_epoch_offset(epoch, config)
        i = max(0, offset - context)

        ov_scale = min(1.0, float(epoch + 1) / max(1, int(config.overlap_warmup_epochs)))
        lambda_ov = float(config.lambda_overlap) * ov_scale

        while i < n:
            start = i
            end = min(i + batch_size + 2 * context, n)
            cs = min(i + context, end)
            ce = min(cs + batch_size, end)
            Bc = ce - cs
            if Bc < 2 or end - start < 2:
                break

            X_full = X_all_perm[start:end]
            with torch.amp.autocast("cuda", enabled=use_amp):
                Y_full = model(X_full)
            Y_full = Y_full.float()
            emb_clamp = 500.0
            Y_full = emb_clamp * torch.tanh(Y_full / emb_clamp)

            P_center = P_cache_gpu.get((int(cs), int(Bc)))
            if P_center is None:
                i += stride
                continue
            # Move to GPU just-in-time if P cache is on CPU
            if not P_center.is_cuda and str(device).startswith("cuda"):
                P_center = P_center.to(device)

            Y_center = Y_full[cs - start:ce - start]
            L_kl = criterion(P_center, Y_center)
            L_temp = temporal_smoothness_loss(Y_full, acf_w)

            L_ov = Y_full.new_tensor(0.0)
            if prev_Y is not None:
                os_ = max(start, int(prev_start))
                oe_ = min(end, int(prev_end))
                if oe_ - os_ >= 2:
                    teacher = prev_Y[os_ - int(prev_start):oe_ - int(prev_start)].detach()
                    student = Y_full[os_ - start:oe_ - start]
                    L_ov = F.mse_loss(student, teacher)

            loss = L_kl + float(config.lambda_temporal) * L_temp + lambda_ov * L_ov

            if torch.isnan(loss) or torch.isinf(loss):
                i += stride
                continue

            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            amp_scaler.step(optimizer)
            amp_scaler.update()

            epoch_loss += loss.item()
            steps += 1
            prev_start, prev_end = start, end
            prev_Y = Y_full.detach()
            i += stride

        if steps == 0:
            continue
        scheduler.step()
        avg = epoch_loss / steps

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("  Epoch %d/%d | KL+Temp Loss: %.4f | steps=%d" % (epoch + 1, int(config.max_epochs), avg, steps))

        if stopper.step(avg, model):
            print("  Early stop epoch %d (best=%.4f)" % (epoch + 1, stopper.best))
            break
        if out_model_name:
            torch.save(model.state_dict(), out_model_name)

    if stopper.state is not None:
        model.load_state_dict(stopper.state)
    model.eval()
    return model


def train_wrapper_safe(model, X_train, path, config, device, layer_name=None,
                       seed=42, retry=0, stage_index=0):
    trained = train_unsupervised_dr(model, X_train, path, config, device, layer_name, stage_index)
    trained.eval()
    n = X_train.shape[0]
    X_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    with torch.no_grad():
        embs = []
        for i in range(0, n, 256):
            embs.append(trained(X_t[i:min(i + 256, n)].permute(0, 3, 1, 2)).cpu())
        emb = torch.cat(embs, dim=0)
    collapsed, reason = check_embedding_collapse(emb)
    if collapsed and retry < int(config.max_retries):
        new_seed = seed + retry * 111 + 42
        print("  [COLLAPSE] %s - retry %d seed=%d" % (reason, retry + 1, new_seed))
        set_all_seeds(new_seed)
        def init_w(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        model.apply(init_w)
        return train_wrapper_safe(model, X_train, path, config, device, layer_name, new_seed, retry + 1, stage_index)
    return trained


def multi_stage_dr(dr_model, X_proj, config, outer, save_dir, device, basename):
    """4-stage DR training: raw -> Dense1 -> Dense2 -> Dense3. ALL UNSUPERVISED."""
    saved = []
    p1 = os.path.join(save_dir, "%s_o%d_s1.pth" % (basename, outer))
    dr_model = train_wrapper_safe(dr_model, X_proj, p1, config, device, None, config.seed, stage_index=0)
    saved.append(p1)

    for si, layer in enumerate(["Dense1", "Dense2", "Dense3"], 1):
        if int(config.n_recur) <= si:
            break
        if os.path.exists(saved[-1]):
            dr_model.load_state_dict(torch.load(saved[-1], map_location=device))
        p = os.path.join(save_dir, "%s_o%d_s%d.pth" % (basename, outer, si + 1))
        dr_model = train_wrapper_safe(dr_model, X_proj, p, config, device, layer, config.seed, stage_index=si)
        saved.append(p)

    return dr_model, saved


def generate_trajectory_for_subject(subj_data, pretrained_encoder, config, device, output_dir):
    """
    Phase B: Generate PURELY UNSUPERVISED trajectory for one subject.
    1) Refine features with pretrained ChebGCN encoder
    2) Run multi-stage parametric t-SNE (KL + temporal ONLY)
    """
    subj_id = subj_data["subject"]
    features = subj_data["features"]

    print("\n" + "=" * 70)
    print("[TRAJECTORY] %s (T=%d)" % (subj_id, features.shape[0]))
    print("=" * 70)

    # Step 1: Refine with pretrained encoder
    refined_3d = refine_with_pretrained_chebgcn(features, pretrained_encoder, config, device)

    # Step 2: Prepare for DR
    T, H, W = refined_3d.shape
    X_proj = refined_3d[..., np.newaxis]  # (T, H, W, 1)
    input_shape = (H, W)

    # Truncate for batch alignment
    X_train, batch_size, n = eeg_data_para(
        X_proj.reshape(T, -1), int(config.balance_degree), int(config.max_batch_size)
    )
    config.batch_size = int(batch_size)
    X_proj_trunc = X_proj[:n]

    print("[DR] T=%d -> truncated=%d, batch_size=%d" % (T, n, batch_size))

    # Step 3: Co-training outer iterations
    # NOTE: Unlike the earlier implementation which retrained ChebGCN each outer iteration,
    # our pretrained encoder has fixed input dimensions. So we always refine
    # from ORIGINAL features. The outer iterations improve the DR model by
    # re-initialising and retraining on the same refined representation.
    save_dir = os.path.join(output_dir, "models", subj_id)
    os.makedirs(save_dir, exist_ok=True)
    all_saved = []
    final_shape = input_shape
    final_ref3d = refined_3d[:n]

    for outer in range(1, int(config.n_iterations) + 1):
        print("\n[Outer %d/%d]" % (outer, int(config.n_iterations)))

        if outer > 1:
            # Re-refine from ORIGINAL features (not DR output)
            # Pretrained encoder has fixed in_channels, so input must match
            ref3d_new = refine_with_pretrained_chebgcn(
                features[:n], pretrained_encoder, config, device
            )
            final_ref3d = ref3d_new
            T2, H2, W2 = ref3d_new.shape
            X_proj_iter = ref3d_new[..., np.newaxis]
            final_shape = (H2, W2)
        else:
            X_proj_iter = X_proj_trunc
            final_shape = input_shape

        dr_model = create_model(
            input_shape=final_shape,
            num_conv_layers=int(config.num_conv_layers),
            filters_list=config.filters_list,
            kernel_size=int(config.kernel_size),
            alpha=float(config.alpha),
            dense_units=config.dense_units,
            final_units=int(config.final_units),
        )

        dr_model, saved = multi_stage_dr(dr_model, X_proj_iter, config, outer, save_dir, device, subj_id)
        all_saved.extend(saved)

    # Step 4: Extract final embedding
    model_path = all_saved[-1]
    X_final = final_ref3d[..., np.newaxis]
    X_t = torch.as_tensor(X_final, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    final_model = create_model(
        input_shape=final_shape,
        num_conv_layers=int(config.num_conv_layers),
        filters_list=config.filters_list,
        kernel_size=int(config.kernel_size),
        alpha=float(config.alpha),
        dense_units=config.dense_units,
        final_units=int(config.final_units),
    )
    final_model.load_state_dict(torch.load(model_path, map_location=device))
    final_model.to(device).eval()

    with torch.no_grad():
        embs = []
        emb_clamp = 500.0
        for i in range(0, len(X_t), 256):
            Y = final_model(X_t[i:i + 256])
            Y = emb_clamp * torch.tanh(Y / emb_clamp)
            embs.append(Y.cpu().numpy())
    embedding = np.concatenate(embs, axis=0)

    print("[TRAJECTORY] %s embedding: %s" % (subj_id, str(embedding.shape)))
    return embedding, n


# ============================================================================
# PART 5: EMBEDDING FEATURE EXTRACTION (for warning system)
# ============================================================================
def extract_embedding_features(emb_2d, window_sizes=(10, 30, 60, 120, 300)):
    """Multi-scale features from 2D embedding trajectory."""
    N = emb_2d.shape[0]
    if N < 2:
        return np.zeros((N, 1), dtype=np.float32)

    emb_mu = emb_2d.mean(axis=0, keepdims=True)
    emb_sd = emb_2d.std(axis=0, keepdims=True) + 1e-8
    emb_norm = (emb_2d - emb_mu) / emb_sd

    bl_n = max(10, int(N * 0.10))
    bl_centroid = emb_norm[:bl_n].mean(axis=0)

    dist = np.linalg.norm(emb_norm - bl_centroid, axis=1)
    dx = np.diff(emb_norm, axis=0, prepend=emb_norm[:1])
    velocity = np.linalg.norm(dx, axis=1)
    accel = np.abs(np.diff(velocity, prepend=velocity[:1]))
    angles = np.arctan2(dx[:, 1], dx[:, 0])
    d_angle = np.abs(np.diff(angles, prepend=angles[:1]))
    d_angle = np.minimum(d_angle, 2 * np.pi - d_angle)
    curvature = d_angle / (velocity + 1e-6)

    local_density = np.zeros(N, dtype=np.float32)
    density_win = 50
    for i in range(N):
        w_st = max(0, i - density_win)
        w_ed = min(N, i + density_win)
        local_pts = emb_norm[w_st:w_ed]
        dists_local = np.linalg.norm(local_pts - emb_norm[i], axis=1)
        dists_local.sort()
        local_density[i] = np.mean(dists_local[1:min(11, len(dists_local))])

    feat_cols = [emb_norm[:, 0], emb_norm[:, 1], dist, velocity, accel, d_angle, curvature, local_density]

    for w in window_sizes:
        if w >= N:
            w = max(2, N // 2)
        rm_dist = uniform_filter1d(dist, size=w, mode='nearest')
        rs_dist = np.sqrt(np.maximum(uniform_filter1d(dist ** 2, size=w, mode='nearest') - rm_dist ** 2, 0.0))
        rmax_dist = maximum_filter1d(dist, size=w, mode='nearest')
        rm_vel = uniform_filter1d(velocity, size=w, mode='nearest')
        rs_vel = np.sqrt(np.maximum(uniform_filter1d(velocity ** 2, size=w, mode='nearest') - rm_vel ** 2, 0.0))
        rm_angvel = uniform_filter1d(d_angle, size=w, mode='nearest')
        rm_curv = uniform_filter1d(curvature, size=w, mode='nearest')
        rm_x = uniform_filter1d(emb_norm[:, 0], size=w, mode='nearest')
        rm_y = uniform_filter1d(emb_norm[:, 1], size=w, mode='nearest')
        rolling_disp = np.sqrt((rm_x - bl_centroid[0]) ** 2 + (rm_y - bl_centroid[1]) ** 2)
        feat_cols.extend([rm_dist, rs_dist, rmax_dist, rm_vel, rs_vel, rm_angvel, rm_curv, rolling_disp])

    features = np.column_stack(feat_cols).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# PART 6: CROSS-SUBJECT WARNING EVALUATION
# ============================================================================
def train_warning_model(train_features, train_gt_state, train_gt_risk):
    """Train warning models on training subjects' data."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_features)

    # Risk regressor
    gb_risk = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    gb_risk.fit(X_train_s, train_gt_risk)

    # State classifier with balanced weights
    n_classes = len(np.unique(train_gt_state))
    class_counts = np.bincount(train_gt_state.astype(int), minlength=3).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    class_w = float(np.sum(class_counts)) / (float(n_classes) * class_counts)
    sample_weights = class_w[train_gt_state.astype(int)]

    gb_cls = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    gb_cls.fit(X_train_s, train_gt_state, sample_weight=sample_weights)

    # Optimal threshold
    y_bin = (train_gt_state >= 1).astype(int)
    risk_pred_train = np.clip(gb_risk.predict(X_train_s), 0, 1)
    if len(np.unique(y_bin)) > 1:
        prec, rec, thrs = precision_recall_curve(y_bin, risk_pred_train)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = np.argmax(f1s)
        threshold = float(thrs[min(best_idx, len(thrs) - 1)])
    else:
        threshold = 0.3
    threshold = np.clip(threshold, 0.05, 0.9)

    return {"gb_risk": gb_risk, "gb_cls": gb_cls, "scaler": scaler, "threshold": threshold}


def evaluate_on_subject(models, test_features, test_gt_state, test_gt_risk, test_seg_infos):
    """Evaluate warning models on a held-out subject."""
    scaler = models["scaler"]
    threshold = models["threshold"]

    X_test_s = scaler.transform(test_features)
    risk_pred = np.clip(models["gb_risk"].predict(X_test_s), 0, 1)
    state_pred = models["gb_cls"].predict(X_test_s)
    warning = (risk_pred >= threshold).astype(int)

    results = {"threshold": threshold}

    # Binary AUC
    y_bin = (test_gt_state >= 1).astype(int)
    if len(np.unique(y_bin)) > 1:
        results["risk_auc"] = float(roc_auc_score(y_bin, risk_pred))
    else:
        results["risk_auc"] = float("nan")

    # Risk correlation
    if np.std(test_gt_risk) > 1e-6 and np.std(risk_pred) > 1e-6:
        results["risk_pearson"] = float(pearsonr(test_gt_risk, risk_pred)[0])
    else:
        results["risk_pearson"] = 0.0

    # Classification
    report = classification_report(
        test_gt_state, state_pred, labels=[0, 1, 2],
        target_names=["BL", "TR", "IC"], output_dict=True, zero_division=0,
    )
    results["weighted_f1"] = float(report["weighted avg"]["f1-score"])
    results["macro_f1"] = float(report["macro avg"]["f1-score"])
    for cls in ["BL", "TR", "IC"]:
        results["%s_f1" % cls] = float(report[cls]["f1-score"])
        results["%s_recall" % cls] = float(report[cls]["recall"])

    # Per-seizure detection
    seizure_detections = []
    false_alarms = []
    N = len(test_gt_state)

    for s in test_seg_infos:
        st, ed = int(s["start"]), min(int(s["end"]), N)
        if ed <= st:
            continue
        seg_risk = risk_pred[st:ed]
        seg_warn = warning[st:ed]
        seg_gt = test_gt_state[st:ed]

        if s["has_seizure"]:
            ictal_idx = np.where(seg_gt == 2)[0]
            if len(ictal_idx) > 0:
                onset = int(ictal_idx[0])
                pre_warn = np.where(seg_warn[:onset] == 1)[0]
                if len(pre_warn) > 0:
                    lead_time = float(onset - int(pre_warn[0]))
                    detected = True
                else:
                    lead_time = 0.0
                    detected = bool(np.any(seg_warn[:onset + 30] == 1))
                seizure_detections.append({
                    "file": s["file"], "detected": detected,
                    "lead_time_sec": lead_time,
                    "max_risk_pre_onset": float(np.max(seg_risk[:max(1, onset)])),
                })
        else:
            fa_rate = float(np.mean(seg_warn)) if len(seg_warn) > 0 else 0.0
            false_alarms.append({"file": s["file"], "fa_rate": fa_rate})

    results["seizure_detections"] = seizure_detections
    results["false_alarms"] = false_alarms
    results["risk_pred"] = risk_pred
    results["state_pred"] = state_pred

    if seizure_detections:
        results["detection_rate"] = float(np.mean([d["detected"] for d in seizure_detections]))
        detected_lt = [d["lead_time_sec"] for d in seizure_detections if d["detected"] and d["lead_time_sec"] > 0]
        results["mean_lead_time"] = float(np.mean(detected_lt)) if detected_lt else 0.0
    else:
        results["detection_rate"] = float("nan")
        results["mean_lead_time"] = float("nan")

    if false_alarms:
        results["mean_fa_rate"] = float(np.mean([f["fa_rate"] for f in false_alarms]))
    else:
        results["mean_fa_rate"] = float("nan")

    return results


def run_leave_one_subject_out(all_subject_results, output_dir):
    """
    Leave-one-subject-out cross-validation for the warning system.
    Train on N-1 subjects' embedding features, test on held-out subject.
    """
    n_subjects = len(all_subject_results)
    print("\n" + "=" * 70)
    print("[LOSO] Leave-One-Subject-Out: %d subjects" % n_subjects)
    print("=" * 70)

    all_loso_results = []

    for test_idx in range(n_subjects):
        test_subj = all_subject_results[test_idx]
        test_id = test_subj["subject"]

        # Gather training data from all other subjects
        train_features_list = []
        train_gt_state_list = []
        train_gt_risk_list = []

        for train_idx in range(n_subjects):
            if train_idx == test_idx:
                continue
            s = all_subject_results[train_idx]
            train_features_list.append(s["emb_features"])
            train_gt_state_list.append(s["gt_state"])
            train_gt_risk_list.append(s["gt_risk"])

        if not train_features_list:
            continue

        train_features = np.concatenate(train_features_list, axis=0)
        train_gt_state = np.concatenate(train_gt_state_list, axis=0)
        train_gt_risk = np.concatenate(train_gt_risk_list, axis=0)

        print("\n[LOSO] Test: %s | Train: %d subjects, %d samples"
              % (test_id, n_subjects - 1, len(train_gt_state)))

        # Train warning model
        models = train_warning_model(train_features, train_gt_state, train_gt_risk)

        # Evaluate on test subject
        results = evaluate_on_subject(
            models,
            test_subj["emb_features"],
            test_subj["gt_state"],
            test_subj["gt_risk"],
            test_subj["seg_infos"],
        )
        results["test_subject"] = test_id

        print("  AUC=%.3f | F1=%.3f | Det=%.1f%% | Lead=%.0fs | FA=%.1f%%"
              % (results.get("risk_auc", 0), results.get("weighted_f1", 0),
                 results.get("detection_rate", 0) * 100,
                 results.get("mean_lead_time", 0),
                 results.get("mean_fa_rate", 0) * 100))

        all_loso_results.append(results)

    return all_loso_results


def run_within_subject_evaluation(subj_result, train_ratio=0.2):
    """
    Within-subject temporal split evaluation.
    Train on first train_ratio of data, test on rest.
    This complements LOSO with subject-specific evaluation.
    """
    features = subj_result["emb_features"]
    gt_state = subj_result["gt_state"]
    gt_risk = subj_result["gt_risk"]
    seg_infos = subj_result["seg_infos"]
    N = len(features)

    n_train = max(100, int(N * train_ratio))

    print("[WITHIN-SUBJ] %s: train=%d, test=%d" % (subj_result["subject"], n_train, N - n_train))

    models = train_warning_model(features[:n_train], gt_state[:n_train], gt_risk[:n_train])

    # Evaluate on test portion
    test_seg_infos = []
    for s in seg_infos:
        st, ed = int(s["start"]), int(s["end"])
        if st >= n_train:
            s_new = dict(s)
            s_new["start"] = st - n_train
            s_new["end"] = ed - n_train
            test_seg_infos.append(s_new)
        elif ed > n_train:
            s_new = dict(s)
            s_new["start"] = 0
            s_new["end"] = ed - n_train
            test_seg_infos.append(s_new)

    results = evaluate_on_subject(
        models, features[n_train:], gt_state[n_train:], gt_risk[n_train:], test_seg_infos
    )
    results["n_train"] = n_train
    results["models"] = models
    results["risk_pred_all"] = np.concatenate([
        np.clip(models["gb_risk"].predict(models["scaler"].transform(features[:n_train])), 0, 1),
        results["risk_pred"]
    ])
    return results


# ============================================================================
# PART 7: VISUALISATION
# ============================================================================
def plot_full_trajectory(emb_2d, gt_state, seg_infos, output_dir, subj_id):
    """Plot trajectory colored by state and time."""
    os.makedirs(output_dir, exist_ok=True)
    N = len(emb_2d)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    ax = axes[0]
    for si, (color, label) in enumerate(zip([HEALTHY, TRANSITION, DISEASE], ["Baseline", "Transition", "Ictal"])):
        mask = gt_state == si
        if np.any(mask):
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, s=2, alpha=0.4, label=label, rasterized=True)
    ax.set_xlabel("Embedding dim 1", fontsize=12)
    ax.set_ylabel("Embedding dim 2", fontsize=12)
    ax.set_title("%s - Neural Manifold (by State) [UNSUPERVISED]" % subj_id, fontsize=14)
    ax.legend(fontsize=11, markerscale=5)

    ax = axes[1]
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(N), cmap='viridis', s=2, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Time (s)", shrink=0.8)
    ax.set_xlabel("Embedding dim 1", fontsize=12)
    ax.set_ylabel("Embedding dim 2", fontsize=12)
    ax.set_title("%s - Neural Manifold (by Time)" % subj_id, fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "%s_trajectory.png" % subj_id), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_seizure_trajectories(emb_2d, seg_infos, output_dir, subj_id):
    """Per-seizure-file zoom plots."""
    seizure_segs = [s for s in seg_infos if s["has_seizure"]]
    if not seizure_segs:
        return

    n_seiz = len(seizure_segs)
    cols = min(3, n_seiz)
    rows = (n_seiz + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    for idx, s in enumerate(seizure_segs):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        st, ed = int(s["start"]), int(s["end"])
        seg_emb = emb_2d[st:ed]
        seg_gt = s["gt_state"][:ed - st]

        for si, (color, label) in enumerate(zip([HEALTHY, TRANSITION, DISEASE], ["BL", "TR", "IC"])):
            mask = seg_gt == si
            if np.any(mask):
                ax.scatter(seg_emb[mask, 0], seg_emb[mask, 1], c=color, s=8, alpha=0.6, label=label)
        ax.plot(seg_emb[:, 0], seg_emb[:, 1], 'k-', alpha=0.15, linewidth=0.5)
        ax.set_title(s["file"], fontsize=11)
        ax.legend(fontsize=9, markerscale=2)

    for idx in range(n_seiz, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle("%s - Seizure Trajectories [UNSUPERVISED]" % subj_id, fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "%s_seizure_trajectories.png" % subj_id), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_biomarker_timeline(emb_2d, gt_state, gt_risk, seg_infos, warn_results, output_dir, subj_id):
    """Biomarker timeline with train/test split."""
    os.makedirs(output_dir, exist_ok=True)
    N = len(emb_2d)
    n_train = int(warn_results.get("n_train", int(N * 0.2)))
    risk_pred = warn_results.get("risk_pred_all", np.zeros(N))
    threshold = float(warn_results.get("threshold", 0.3))

    bl_n = max(10, int(N * 0.10))
    bl_centroid = emb_2d[:bl_n].mean(axis=0)
    dist = np.linalg.norm(emb_2d - bl_centroid, axis=1)
    dist_smooth = gaussian_filter1d(dist, sigma=10)
    risk_smooth = gaussian_filter1d(risk_pred, sigma=10)

    t = np.arange(N)
    fig, axes = plt.subplots(4, 1, figsize=(24, 14), sharex=True,
                              gridspec_kw={'height_ratios': [1, 2, 2, 1]})

    ax = axes[0]
    for si, (color, label) in enumerate(zip([HEALTHY, TRANSITION, DISEASE], ["Baseline", "Transition", "Ictal"])):
        mask = gt_state == si
        if np.any(mask):
            ax.fill_between(t, 0, 1, where=mask, color=color, alpha=0.7, label=label)
    ax.axvline(n_train, color='black', linestyle='--', linewidth=1.5, label='Train/Test')
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("%s - Biomarker Timeline [UNSUPERVISED trajectory]" % subj_id, fontsize=14)
    ax.legend(loc='upper right', fontsize=9, ncol=4)

    ax = axes[1]
    ax.plot(t, dist_smooth, color=PALETTE["distance"], linewidth=0.8, alpha=0.9)
    ax.fill_between(t, 0, dist_smooth, alpha=0.15, color=PALETTE["distance"])
    ax.axvline(n_train, color='black', linestyle='--', linewidth=1.5)
    ax.set_ylabel("Manifold distance", fontsize=11)

    ax = axes[2]
    ax.plot(t, risk_smooth, color=PALETTE["accent"], linewidth=0.8, label="Predicted risk")
    ax.plot(t, gt_risk, color='gray', linewidth=0.5, alpha=0.5, label="GT risk")
    ax.axhline(threshold, color=PALETTE["threshold"], linestyle='--', linewidth=1.2, label="Threshold")
    ax.axvline(n_train, color='black', linestyle='--', linewidth=1.5)
    ax.set_ylabel("Risk", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[3]
    warning_full = np.zeros(N)
    warning_full[n_train:] = (risk_pred[n_train:] >= threshold).astype(float)
    ax.fill_between(t, 0, warning_full, color=PALETTE["accent"], alpha=0.5)
    ax.axvline(n_train, color='black', linestyle='--', linewidth=1.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OFF", "ON"])
    ax.set_ylabel("Warning", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "%s_biomarker_timeline.png" % subj_id), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_warning_evaluation(warn_results, gt_state, gt_risk, output_dir, subj_id):
    """ROC, confusion matrix, detection summary."""
    os.makedirs(output_dir, exist_ok=True)
    n_train = int(warn_results.get("n_train", 0))
    risk_pred = warn_results["risk_pred"]
    state_pred = warn_results["state_pred"]
    y_state = gt_state[n_train:] if n_train > 0 else gt_state
    y_risk = gt_risk[n_train:] if n_train > 0 else gt_risk

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ROC
    ax = fig.add_subplot(gs[0, 0])
    y_bin = (y_state >= 1).astype(int)
    if len(np.unique(y_bin)) > 1:
        fpr, tpr, _ = roc_curve(y_bin, risk_pred)
        auc_val = roc_auc_score(y_bin, risk_pred)
        ax.plot(fpr, tpr, color=PALETTE["accent"], linewidth=2, label="AUC=%.3f" % auc_val)
        ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["roc_fill"])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_title("ROC: At-Risk Detection", fontsize=12)
    ax.legend(fontsize=11)

    # Risk scatter
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(y_risk, risk_pred, c='steelblue', s=1, alpha=0.2, rasterized=True)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax.set_title("Risk Prediction (r=%.3f)" % warn_results.get("risk_pearson", 0), fontsize=12)

    # Confusion matrix
    ax = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_state, state_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.2f\n(%d)" % (cm_norm[i, j], cm[i, j]),
                    ha='center', va='center', fontsize=9, color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["BL", "TR", "IC"])
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["BL", "TR", "IC"])
    ax.set_title("Confusion Matrix", fontsize=12)

    # Seizure detections
    ax = fig.add_subplot(gs[1, 0])
    detections = warn_results.get("seizure_detections", [])
    if detections:
        files = [d["file"].split("_")[-1].replace(".edf", "") for d in detections]
        lt = [d["lead_time_sec"] for d in detections]
        colors = [PALETTE["lead_time"] if d["detected"] else PALETTE["accent"] for d in detections]
        ax.barh(range(len(files)), lt, color=colors)
        ax.set_yticks(range(len(files)))
        ax.set_yticklabels(files, fontsize=9)
        ax.set_xlabel("Lead Time (s)")
        ax.set_title("Seizure Detection")
    else:
        ax.text(0.5, 0.5, "No seizures in test", ha='center', va='center', transform=ax.transAxes)

    # False alarms
    ax = fig.add_subplot(gs[1, 1])
    fa_list = warn_results.get("false_alarms", [])
    if fa_list:
        fa_files = [f["file"].split("_")[-1].replace(".edf", "") for f in fa_list]
        fa_rates = [f["fa_rate"] * 100 for f in fa_list]
        colors = [PALETTE["accent"] if r > 5 else PALETTE["lead_time"] for r in fa_rates]
        ax.barh(range(len(fa_files)), fa_rates, color=colors)
        ax.set_yticks(range(len(fa_files)))
        ax.set_yticklabels(fa_files, fontsize=7)
        ax.set_xlabel("FA Rate (%)")
        ax.set_title("False Alarms")

    plt.suptitle("%s - Warning Evaluation [UNSUPERVISED trajectory + Supervised warning]" % subj_id, fontsize=14)
    fig.savefig(os.path.join(output_dir, "%s_warning_eval.png" % subj_id), dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_loso_summary(loso_results, output_dir):
    """Plot summary of LOSO cross-validation across all subjects."""
    os.makedirs(output_dir, exist_ok=True)
    if not loso_results:
        return

    subjects = [r["test_subject"] for r in loso_results]
    aucs = [r.get("risk_auc", 0) for r in loso_results]
    det_rates = [r.get("detection_rate", 0) * 100 for r in loso_results]
    fa_rates = [r.get("mean_fa_rate", 0) * 100 for r in loso_results]
    lead_times = [r.get("mean_lead_time", 0) for r in loso_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.barh(range(len(subjects)), aucs, color='steelblue')
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("AUC")
    ax.set_title("Risk AUC (LOSO)")
    ax.axvline(np.nanmean(aucs), color='red', linestyle='--', label="Mean=%.3f" % np.nanmean(aucs))
    ax.legend()

    ax = axes[0, 1]
    valid_det = [d for d in det_rates if not np.isnan(d)]
    ax.barh(range(len(subjects)), det_rates, color=PALETTE["lead_time"])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("Seizure Detection Rate (LOSO)")

    ax = axes[1, 0]
    ax.barh(range(len(subjects)), fa_rates, color=PALETTE["accent"])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("FA Rate (%)")
    ax.set_title("False Alarm Rate (LOSO)")

    ax = axes[1, 1]
    ax.barh(range(len(subjects)), lead_times, color=PALETTE["distance"])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("Lead Time (s)")
    ax.set_title("Mean Lead Time (LOSO)")

    plt.suptitle("Cross-Subject LOSO Evaluation Summary", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "LOSO_summary.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# PART 8: MAIN DRIVER
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Bio-PD EEG Pipeline")
    parser.add_argument("--data_root", type=str, default="./data_chb",
                        help="Root directory containing chb01, chb02, ..., chb24 subdirectories")
    parser.add_argument("--output_dir", type=str, default="./results/biopd_eeg")
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--seed", type=int, default=42)

    # Subject selection
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subject IDs (e.g., 'chb01,chb02'). Default: all")
    parser.add_argument("--test_subject", type=str, default=None,
                        help="Single subject for detailed analysis. Default: all via LOSO")

    # Pretraining
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--pretrain_chunk_size", type=int, default=500)
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Load existing pretrained encoder instead of training")

    # DR training
    parser.add_argument("--n_iterations", type=int, default=2)
    parser.add_argument("--n_recur", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--balance_degree", type=int, default=50)

    # Warning
    parser.add_argument("--train_ratio", type=float, default=0.2,
                        help="Within-subject train ratio for temporal split evaluation")
    parser.add_argument("--skip_loso", action="store_true",
                        help="Skip LOSO, only do within-subject evaluation")
    parser.add_argument("--clear_emb_cache", action="store_true",
                        help="Delete cached embeddings to force re-generation")
    parser.add_argument("--no_timecorr", action="store_true",
                        help="Disable timecorr smoothing in ChebGCN refinement")
    parser.add_argument("--timecorr_radius", type=int, default=30,
                        help="Max smoothing radius in seconds (default 30). Smaller=sharper boundaries, larger=smoother trajectories")
    parser.add_argument("--max_batch_size", type=int, default=4000,
                        help="Max batch size for P matrix (prevents OOM on large subjects)")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = ConfigCrossSubject()
    config.seed = args.seed
    config.pretrain_epochs = args.pretrain_epochs
    config.pretrain_chunk_size = args.pretrain_chunk_size
    config.n_iterations = args.n_iterations
    config.n_recur = args.n_recur
    config.max_epochs = args.max_epochs
    config.balance_degree = args.balance_degree
    config.max_batch_size = args.max_batch_size
    config.timecorr_max_radius = args.timecorr_radius
    if args.no_timecorr:
        config.use_timecorr_in_chunks = False

    set_all_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device: %s | GPU: %s" % (str(device), args.gpu))

    output_dir = args.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ================================================================
    # STEP 1: Discover and preprocess all subjects
    # ================================================================
    print("\n" + "=" * 80)
    print("[STEP 1] Discover subjects")
    print("=" * 80)

    all_subjects = discover_all_subjects(args.data_root)
    if not all_subjects:
        raise RuntimeError("No subjects found in %s" % args.data_root)

    if args.subjects:
        selected = [s.strip() for s in args.subjects.split(",")]
        all_subjects = [s for s in all_subjects if s["id"] in selected]
        print("  Selected %d subjects: %s" % (len(all_subjects), str(selected)))

    print("\n[STEP 1b] Preprocessing all subjects ...")
    all_subject_data = []
    for subj_info in all_subjects:
        cache_path = os.path.join(output_dir, "cache", "%s_preprocessed.pkl" % subj_info["id"])
        if os.path.exists(cache_path):
            print("  Loading cached: %s" % subj_info["id"])
            with open(cache_path, "rb") as f:
                subj_data = pickle.load(f)
        else:
            subj_data = preprocess_subject(subj_info)
            if subj_data is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(subj_data, f)

        if subj_data is not None:
            all_subject_data.append(subj_data)
        gc.collect()

    print("\n[STEP 1] %d subjects preprocessed successfully" % len(all_subject_data))

    # ================================================================
    # STEP 2: Cross-subject ChebGCN pretraining (self-supervised)
    # ================================================================
    print("\n" + "=" * 80)
    print("[STEP 2] Cross-Subject ChebGCN Pretraining (SELF-SUPERVISED, no labels)")
    print("=" * 80)

    pretrain_path = os.path.join(model_dir, "pretrained_chebgcn.pth")
    pretrained_in_ch = None

    if args.skip_pretrain and os.path.exists(pretrain_path):
        print("  Loading pretrained encoder from %s" % pretrain_path)
        # Need to determine in_ch
        sample = all_subject_data[0]
        ch_feat = _reorganize_features_by_channel(sample["features"][:10], int(config.n_channels), 9999)
        pretrained_in_ch = ch_feat.shape[2]
        pretrained_encoder = SharedChebGCNEncoder(
            pretrained_in_ch, int(config.cheb_hidden_dim), int(config.cheb_out_channels), int(config.cheb_K)
        ).to(device)
        pretrained_encoder.load_state_dict(torch.load(pretrain_path, map_location=device))
    else:
        pretrained_encoder, pretrained_in_ch = pretrain_cross_subject_chebgcn(
            all_subject_data, config, device, pretrain_path
        )

    # ================================================================
    # STEP 3: Generate UNSUPERVISED trajectories for each subject
    # ================================================================
    print("\n" + "=" * 80)
    print("[STEP 3] Generate Unsupervised Trajectories (KL + Temporal ONLY, no labels)")
    print("=" * 80)

    all_subject_results = []

    # Clear embedding cache if requested (needed after changing timecorr settings)
    if args.clear_emb_cache:
        cache_dir = os.path.join(output_dir, "cache")
        cleared = 0
        for f in glob.glob(os.path.join(cache_dir, "*_embedding.npy")) + \
                 glob.glob(os.path.join(cache_dir, "*_n_truncated.npy")):
            os.remove(f)
            cleared += 1
        print("  [CACHE] Cleared %d embedding cache files" % cleared)

    for subj_data in all_subject_data:
        subj_id = subj_data["subject"]

        emb_cache = os.path.join(output_dir, "cache", "%s_embedding.npy" % subj_id)
        n_cache = os.path.join(output_dir, "cache", "%s_n_truncated.npy" % subj_id)

        if os.path.exists(emb_cache) and os.path.exists(n_cache):
            print("  Loading cached embedding: %s" % subj_id)
            embedding = np.load(emb_cache)
            n_truncated = int(np.load(n_cache))
        else:
            embedding, n_truncated = generate_trajectory_for_subject(
                subj_data, pretrained_encoder, config, device,
                os.path.join(output_dir, "models")
            )
            os.makedirs(os.path.dirname(emb_cache), exist_ok=True)
            np.save(emb_cache, embedding)
            np.save(n_cache, np.array(n_truncated))

        # Truncate everything to match
        gt_state = subj_data["gt_state"][:n_truncated]
        gt_risk = subj_data["gt_risk"][:n_truncated]
        seg_infos = []
        for s in subj_data["seg_infos"]:
            st, ed = int(s["start"]), int(s["end"])
            if st >= n_truncated:
                continue
            ed_c = min(ed, n_truncated)
            if ed_c - st < 10:
                continue
            s_new = dict(s)
            s_new["end"] = ed_c
            s_new["gt_state"] = s["gt_state"][:ed_c - st]
            s_new["gt_risk"] = s["gt_risk"][:ed_c - st]
            seg_infos.append(s_new)

        # Z-score normalise embedding
        emb_2d = embedding[:, :2]
        emb_mu = emb_2d.mean(axis=0, keepdims=True)
        emb_sd = emb_2d.std(axis=0, keepdims=True) + 1e-8
        emb_2d_norm = (emb_2d - emb_mu) / emb_sd

        # Extract embedding features
        emb_features = extract_embedding_features(emb_2d_norm)

        all_subject_results.append({
            "subject": subj_id,
            "embedding": emb_2d,
            "embedding_norm": emb_2d_norm,
            "emb_features": emb_features,
            "gt_state": gt_state,
            "gt_risk": gt_risk,
            "seg_infos": seg_infos,
            "n_truncated": n_truncated,
        })

        # Plot trajectories
        plot_full_trajectory(emb_2d, gt_state, seg_infos, plot_dir, subj_id)
        plot_seizure_trajectories(emb_2d, seg_infos, plot_dir, subj_id)

        print("  %s: emb=%s, features=%s, BL=%d TR=%d IC=%d"
              % (subj_id, str(emb_2d.shape), str(emb_features.shape),
                 int(np.sum(gt_state == 0)), int(np.sum(gt_state == 1)), int(np.sum(gt_state == 2))))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================================================
    # STEP 4: Warning evaluation
    # ================================================================
    print("\n" + "=" * 80)
    print("[STEP 4] Supervised Warning Evaluation")
    print("=" * 80)

    # 4a: Leave-one-subject-out (cross-subject generalisation)
    if not args.skip_loso and len(all_subject_results) > 2:
        loso_results = run_leave_one_subject_out(all_subject_results, output_dir)
        plot_loso_summary(loso_results, plot_dir)

        # Print LOSO summary
        print("\n" + "-" * 60)
        print("[LOSO SUMMARY]")
        aucs = [r.get("risk_auc", float("nan")) for r in loso_results]
        dets = [r.get("detection_rate", float("nan")) for r in loso_results]
        fas = [r.get("mean_fa_rate", float("nan")) for r in loso_results]
        lts = [r.get("mean_lead_time", float("nan")) for r in loso_results]
        print("  Mean AUC:        %.3f +/- %.3f" % (np.nanmean(aucs), np.nanstd(aucs)))
        print("  Mean Detection:  %.1f%% +/- %.1f%%" % (np.nanmean(dets) * 100, np.nanstd(dets) * 100))
        print("  Mean FA Rate:    %.1f%% +/- %.1f%%" % (np.nanmean(fas) * 100, np.nanstd(fas) * 100))
        print("  Mean Lead Time:  %.0f +/- %.0f sec" % (np.nanmean(lts), np.nanstd(lts)))
        print("-" * 60)
    else:
        loso_results = None
        print("  [SKIP] LOSO (need > 2 subjects)")

    # 4b: Within-subject temporal evaluation (for detailed per-subject analysis)
    print("\n[STEP 4b] Within-Subject Temporal Evaluation")
    within_results = {}

    subjects_to_evaluate = all_subject_results
    if args.test_subject:
        subjects_to_evaluate = [s for s in all_subject_results if s["subject"] == args.test_subject]

    for subj_result in subjects_to_evaluate:
        subj_id = subj_result["subject"]
        n_seiz = sum(1 for s in subj_result["seg_infos"] if s["has_seizure"])
        if n_seiz == 0:
            print("  [SKIP] %s: no seizures" % subj_id)
            continue

        ws_result = run_within_subject_evaluation(subj_result, train_ratio=args.train_ratio)

        print("  %s: AUC=%.3f Det=%.1f%% Lead=%.0fs FA=%.1f%%"
              % (subj_id,
                 ws_result.get("risk_auc", 0),
                 ws_result.get("detection_rate", 0) * 100,
                 ws_result.get("mean_lead_time", 0),
                 ws_result.get("mean_fa_rate", 0) * 100))

        # Detailed plots for this subject
        plot_biomarker_timeline(
            subj_result["embedding_norm"], subj_result["gt_state"], subj_result["gt_risk"],
            subj_result["seg_infos"], ws_result, plot_dir, subj_id
        )
        plot_warning_evaluation(
            ws_result, subj_result["gt_state"], subj_result["gt_risk"], plot_dir, subj_id
        )

        within_results[subj_id] = ws_result

    # ================================================================
    # STEP 5: Save summary
    # ================================================================
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Bio-PD EEG Pipeline - Summary\n")
        f.write("=" * 60 + "\n")
        f.write("Date: %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("Data root: %s\n" % args.data_root)
        f.write("Output: %s\n" % output_dir)
        f.write("Subjects: %d\n\n" % len(all_subject_results))

        f.write("DESIGN PRINCIPLES:\n")
        f.write("  1. Trajectory generation: PURELY UNSUPERVISED (KL + temporal only)\n")
        f.write("  2. ChebGCN pretraining: SELF-SUPERVISED (reconstruction loss)\n")
        f.write("  3. Warning system: SUPERVISED with proper evaluation\n")
        f.write("     - LOSO cross-validation (cross-subject generalisation)\n")
        f.write("     - Within-subject temporal split (subject-specific)\n")
        f.write("  4. NO label leakage into trajectory generation\n\n")

        if loso_results:
            f.write("LOSO Cross-Validation:\n")
            for r in loso_results:
                f.write("  %s: AUC=%.3f Det=%.1f%% FA=%.1f%% Lead=%.0fs\n"
                        % (r["test_subject"],
                           r.get("risk_auc", 0),
                           r.get("detection_rate", 0) * 100,
                           r.get("mean_fa_rate", 0) * 100,
                           r.get("mean_lead_time", 0)))
            f.write("\n")

        f.write("Within-Subject Results:\n")
        for subj_id, r in within_results.items():
            f.write("  %s: AUC=%.3f Det=%.1f%% FA=%.1f%% Lead=%.0fs\n"
                    % (subj_id,
                       r.get("risk_auc", 0),
                       r.get("detection_rate", 0) * 100,
                       r.get("mean_fa_rate", 0) * 100,
                       r.get("mean_lead_time", 0)))
        f.write("\n")

        f.write("Configuration:\n")
        for attr in ["pretrain_epochs", "pretrain_chunk_size", "n_iterations", "n_recur",
                     "max_epochs", "balance_degree", "learning_rate", "lambda_temporal",
                     "perplexity", "cheb_hidden_dim", "cheb_out_channels"]:
            f.write("  %-25s %s\n" % (attr, str(getattr(config, attr, "?"))))

    print("\n[SAVED] Summary: %s" % summary_path)
    print("[SAVED] Plots: %s" % plot_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
