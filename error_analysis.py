"""
This file contains all necessary functions to perform an error analysis in the main simulations.
"""

import numpy as np
from collections import deque
from scipy.optimize import curve_fit
import warnings

def acf(data, max_lag):
    """
    Computes the normalized autocorrelation function of a time series up to a maximum lag.

    Parameters
    ----------
    data : np.ndarray
        The input time series data.
    max_lag : int
        Maximum lag value to compute autocorrelation for.

    Returns
    -------
    np.ndarray
        Autocorrelation values for lags from 1 to max_lag.
    """
    data = np.asarray(data)
    N = len(data)
    mean = np.mean(data)
    var = np.var(data)
    autocorr_func = np.zeros(max_lag)

    for t in range(1, max_lag + 1):  # 1 ≤ n ≤ N − t
        num = (N - t) * np.sum(data[:-t] * data[t:]) - np.sum(data[:-t]) * np.sum(data[t:])
        den1 = np.sqrt((N - t) * np.sum(data[:-t] ** 2) - np.sum(data[:-t]) ** 2)
        den2 = np.sqrt((N - t) * np.sum(data[t:] ** 2) - np.sum(data[t:]) ** 2)

        if den1 == 0 or den2 == 0:  # Avoid division by zero
            autocorr_func[t - 1] = 0
        else:
            autocorr_func[t - 1] = num / (den1 * den2)

    return autocorr_func

def acf_window(acf, c=5.0):
    """
    Determines the maximum lag to use in autocorrelation integration 
    by identifying where the sum of the ACF becomes statistically noisy.

    Parameters
    ----------
    acf : np.ndarray
        Autocorrelation function values.
    c : float, optional
        Threshold multiplier for defining the window cutoff (default is 5.0).

    Returns
    -------
    int
        Maximum lag index to use for integrated autocorrelation time.
    """
    taus = np.cumsum(acf) + 0.5
    for M in range(len(acf)): 
        if M >= c * taus[M]:    # Sokal’s criterion: stop where M ≥ c * τ_int(M)
            return M

    # Warn if we reach the fallback
    warnings.warn(
        "auto_window fallback triggered: using full ACF window (no cutoff found). "
        "Consider increasing max_lag or reducing c.",
        RuntimeWarning
    )

    return len(acf) - 1

def iat_error(data, max_lag=500, c=5.0):
    """
    Estimates the integrated autocorrelation time τ_int using Sokal's method.

    This method integrates the normalized autocorrelation function up to a cutoff
    determined by the auto-windowing rule, which balances bias and variance in τ estimation.

    Parameters
    ----------
    data : np.ndarray
        Input time series data.
    max_lag : int, optional
        Maximum lag for the autocorrelation function (default is 500).
    c : float, optional
        Window cutoff multiplier (default is 5.0).

    Returns
    -------
    int
        Estimated integrated autocorrelation time τ_int.
    error : float
        Estimated standard error of the mean.
    thinned_data : np.ndarray
        Time series thinned by τ_int, assumed approximately independent.
    """
    autocorr_func = acf(data, max_lag)
    autocorr_func /= autocorr_func[0]
    M = acf_window(autocorr_func, c)

    tau_sum = 0.5 + np.sum(autocorr_func[:M])
    if np.isnan(tau_sum) or np.isinf(tau_sum):
        tau_int = 1
    else:
        tau_int = max(1, int(round(0.5 + tau_sum)))
    thinned_data = data[::tau_int]
    
    var_data = np.var(thinned_data)
    error = error = np.sqrt((2 * tau_int / len(data)) * var_data)

    return float(tau_int), error, thinned_data

def data_blocking(data, tau, min_block_size=1):
    """
    Estimates statistical errors using   data blocking method.

    Parameters
    ----------
    data : np.ndarray
        Input correlated time series.
    tau : float
        Estimated autocorrelation time used to set maximum block size.
    min_block_size : int, optional
        Minimum block size to consider (default is 1).

    Returns
    -------
    block_sizes : np.ndarray
        Array of block sizes tested.
    errors : np.ndarray
        Estimated standard error for each block size.
    blocked_data : np.ndarray
        Final block-averaged dataset assumed approximately independent.
    """
    max_block_size = int(np.round(tau*5))   # Ensure it's an integer
    N = len(data)
    
    block_sizes = []
    errors = []

    block_size = min_block_size
    while block_size <= max_block_size and block_size <= N // 2:
        num_blocks = N // block_size  # Only full blocks considered
    
        if num_blocks < 2:
            break  # Avoid division by zero when num_blocks - 1 = 0

        block_sizes.append(block_size)

        # Compute the mean of each block
        blocked_data = data[: num_blocks * block_size].reshape(num_blocks, block_size).mean(axis=1)

        # Compute ⟨a⟩ and ⟨a²⟩
        mean_blocked = np.mean(blocked_data)
        mean_sq_blocked = np.mean(blocked_data**2)

        # Compute error
        error = np.sqrt((mean_sq_blocked - mean_blocked**2) / (num_blocks - 1))
        errors.append(error)

        block_size += 1

    return np.array(block_sizes), np.array(errors), blocked_data

def block_bootstrap(data, num_resamples=1000):
    """
    Estimates uncertainty using the block bootstrap method.

    Resamples the block-averaged (independent) dataset with replacement to 
    generate an empirical distribution of the mean. The spread of this 
    distribution is used to estimate the standard error.

    Parameters
    ----------
    data : np.ndarray
        Block-averaged (independent) input data.
    num_resamples : int
        Number of bootstrap resamples to generate.

    Returns
    -------
    bootstrap_samples : np.ndarray
        Resampled datasets of shape (num_resamples, len(data)).
    """
    N = len(data)  # Number of independent blocks
    bootstrap_samples = np.zeros((num_resamples, N))  # Store resampled datasets

    # Perform bootstrap resampling
    for i in range(num_resamples):
        resampled_data = np.random.choice(data, size=N, replace=True)  # Sample with replacement
        bootstrap_samples[i] = resampled_data

    return bootstrap_samples