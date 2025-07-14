"""
Computational utilities for analyzing adaptability model results.

This module provides computational functions for analyzing, transforming,
and extracting insights from the adaptability model.
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import pandas as pd
import sys
import os

# Add the model directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.adaptability_model import AdaptabilityModel


def compute_adaptability_landscape(model: AdaptabilityModel, 
                                 x_range: Tuple[float, float], 
                                 d_range: Tuple[float, float], 
                                 resolution: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the adaptability landscape A(x,d) over ranges of x and d.
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x_range : Tuple[float, float]
        Range of x values (min, max).
    d_range : Tuple[float, float]
        Range of d values (min, max).
    resolution : Tuple[int, int]
        Resolution (number of points) for x and d.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        x values, d values, and adaptability values A(x,d).
    """
    x_min, x_max = x_range
    d_min, d_max = d_range
    nx, nd = resolution
    
    x_values = np.linspace(x_min, x_max, nx)
    d_values = np.linspace(d_min, d_max, nd)
    
    adaptability_values = np.zeros((nx, nd))
    
    for i, x in enumerate(x_values):
        for j, d in enumerate(d_values):
            adaptability_values[i, j] = model.adaptability(x, d)
    
    return x_values, d_values, adaptability_values


def compute_time_series(model: AdaptabilityModel, 
                       x: float, 
                       d: float, 
                       t_range: Tuple[float, float], 
                       nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the time series of adaptability A(x,d,t) and coherence C(x,d,t).
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x : float
        Current configuration of the system.
    d : float
        Depth parameter.
    t_range : Tuple[float, float]
        Range of time values (min, max).
    nt : int
        Number of time points.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Time values, adaptability values A(x,d,t), and coherence values C(x,d,t).
    """
    t_min, t_max = t_range
    t_values = np.linspace(t_min, t_max, nt)
    
    adaptability_values = np.array([model.adaptability_time(x, d, t) for t in t_values])
    coherence_values = 1 - adaptability_values
    
    return t_values, adaptability_values, coherence_values


def compute_spectral_density(model: AdaptabilityModel, 
                           x: float, 
                           d: float, 
                           t_range: Tuple[float, float], 
                           nt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectral density of adaptability A(x,d,t).
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x : float
        Current configuration of the system.
    d : float
        Depth parameter.
    t_range : Tuple[float, float]
        Range of time values (min, max).
    nt : int
        Number of time points.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency values and power spectral density.
    """
    t_min, t_max = t_range
    t_values = np.linspace(t_min, t_max, nt)
    
    adaptability_values = np.array([model.adaptability_time(x, d, t) for t in t_values])
    
    # Compute the FFT
    fft_values = fft(adaptability_values)
    
    # Compute the frequency values
    sample_spacing = (t_max - t_min) / (nt - 1)
    freq_values = fftfreq(nt, d=sample_spacing)
    
    # Compute the power spectral density
    psd = np.abs(fft_values) ** 2
    
    # Return only the positive frequencies
    positive_freq_mask = freq_values > 0
    
    return freq_values[positive_freq_mask], psd[positive_freq_mask]


def verify_conservation_law(model: AdaptabilityModel, 
                          x_samples: int = 100, 
                          d_samples: int = 10,
                          x_range: Tuple[float, float] = (-1, 1),
                          d_range: Tuple[float, float] = (1, 30)) -> Tuple[np.ndarray, float]:
    """
    Verify the conservation law C(x,d) + A(x,d) = 1 numerically.
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x_samples : int, optional
        Number of x samples, default is 100.
    d_samples : int, optional
        Number of d samples, default is 10.
    x_range : Tuple[float, float], optional
        Range of x values (min, max), default is (-1, 1).
    d_range : Tuple[float, float], optional
        Range of d values (min, max), default is (1, 30).
        
    Returns
    -------
    Tuple[np.ndarray, float]
        The sum of C+A for each sample, and the maximum absolute deviation from 1.
    """
    x_values = np.linspace(x_range[0], x_range[1], x_samples)
    d_values = np.linspace(d_range[0], d_range[1], d_samples)
    
    conservation_sums = np.zeros((x_samples, d_samples))
    
    for i, x in enumerate(x_values):
        for j, d in enumerate(d_values):
            c = model.coherence(x, d)
            a = model.adaptability(x, d)
            conservation_sums[i, j] = c + a
    
    # Calculate max absolute deviation from 1
    max_deviation = np.max(np.abs(conservation_sums - 1))
    
    return conservation_sums, max_deviation


def exponential_decay_function(d, a, b):
    """Exponential decay function: a * exp(b * d)"""
    return a * np.exp(b * d)


def verify_exponential_decay(model: AdaptabilityModel, 
                           x: float, 
                           d_range: Tuple[float, float], 
                           nd: int = 100) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Verify that adaptability A(x,d) decays exponentially with d.
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x : float
        Current configuration of the system.
    d_range : Tuple[float, float]
        Range of d values (min, max).
    nd : int, optional
        Number of d samples, default is 100.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float, float]
        d values, adaptability values, fitted exponent, theoretical exponent, and R² value.
    """
    d_min, d_max = d_range
    d_values = np.linspace(d_min, d_max, nd)
    
    adaptability_values = np.array([model.adaptability(x, d) for d in d_values])
    
    # Calculate theoretical exponent
    m_star, n_star = model.M_star(x)
    theoretical_exponent = -m_star
    
    # Take log of adaptability for non-zero values
    mask = adaptability_values > 0
    log_adapt = np.log(adaptability_values[mask])
    d_masked = d_values[mask]
    
    if len(log_adapt) < 2:
        return d_values, adaptability_values, np.nan, theoretical_exponent, np.nan
    
    # Linear fit to log(adaptability) vs d
    coeffs = np.polyfit(d_masked, log_adapt, 1)
    poly = np.poly1d(coeffs)
    
    # Calculate R² value
    y_fit = poly(d_masked)
    ss_tot = np.sum((log_adapt - np.mean(log_adapt))**2)
    ss_res = np.sum((log_adapt - y_fit)**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # The exponent is the slope of the fit
    exponent = coeffs[0]
    
    return d_values, adaptability_values, exponent, theoretical_exponent, r_squared


def verify_temporal_oscillations(model: AdaptabilityModel, 
                               x: float, 
                               d: float, 
                               t_range: Tuple[float, float], 
                               nt: int = 1000) -> Tuple[float, float, float]:
    """
    Verify that adaptability A(x,d,t) oscillates in time and conserves A+C=1.
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x : float
        Current configuration of the system.
    d : float
        Depth parameter.
    t_range : Tuple[float, float]
        Range of time values (min, max).
    nt : int, optional
        Number of time points, default is 1000.
        
    Returns
    -------
    Tuple[float, float, float]
        Maximum absolute deviation from 1 in A+C, peak-to-peak amplitude of A, and mean A.
    """
    t_values = np.linspace(t_range[0], t_range[1], nt)
    
    # Compute A(x,d,t) and C(x,d,t)
    adaptability_values = np.array([model.adaptability_time(x, d, t) for t in t_values])
    coherence_values = np.array([model.coherence_time(x, d, t) for t in t_values])
    
    # Check conservation law
    conservation_sum = adaptability_values + coherence_values
    max_deviation = np.max(np.abs(conservation_sum - 1))
    
    # Characterize oscillations
    peak_to_peak = np.max(adaptability_values) - np.min(adaptability_values)
    mean_adaptability = np.mean(adaptability_values)
    
    return max_deviation, peak_to_peak, mean_adaptability


def analyze_modal_contributions(model: AdaptabilityModel, 
                              x: float, 
                              d_range: Tuple[float, float], 
                              num_depths: int = 10) -> pd.DataFrame:
    """
    Analyze the relative contributions of different modes to the total adaptability.
    
    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
    x : float
        Current configuration of the system.
    d_range : Tuple[float, float]
        Range of d values (min, max).
    num_depths : int, optional
        Number of depth values to analyze, default is 10.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with relative mode contributions at each depth.
    """
    d_min, d_max = d_range
    d_values = np.linspace(d_min, d_max, num_depths)
    
    # Initialize results storage
    results = []
    
    for d in d_values:
        # Calculate contribution of each mode to adaptability
        mode_contributions = {}
        total_adaptability = 0
        
        for n in model.n_ord:
            contribution = model.coupling_function(x, d, n)
            mode_contributions[f'Mode {n}'] = contribution
            total_adaptability += contribution
        
        # Normalize contributions
        if total_adaptability > 0:
            for mode in mode_contributions:
                mode_contributions[mode] /= total_adaptability * len(model.n_ord)
        
        # Calculate entropy of mode distribution as a measure of complexity
        values = np.array(list(mode_contributions.values()))
        values = values[values > 0]  # Remove zeros
        if len(values) > 0:
            entropy = -np.sum(values * np.log(values)) / np.log(len(values))
        else:
            entropy = 0
        
        mode_contributions['Depth'] = d
        mode_contributions['Total Adaptability'] = model.adaptability(x, d)
        mode_contributions['Normalized Entropy'] = entropy
        
        results.append(mode_contributions)
    
    return pd.DataFrame(results)
def find_mode_crossover(model: AdaptabilityModel,
                         mode_a: int,
                         mode_b: int,
                         x_range: Tuple[float, float] = (-0.5, 0.5),
                         num_points: int = 1000,
                         tol: float = 1e-5) -> List[float]:
    """Find approximate configuration values where two modes have equal decay exponents."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    diff = [model.M_n(x, mode_a) - model.M_n(x, mode_b) for x in x_values]
    crossovers = []
    for i in range(len(diff) - 1):
        if np.isinf(diff[i]) or np.isinf(diff[i+1]):
            continue
        if diff[i] == 0:
            crossovers.append(x_values[i])
        if diff[i] * diff[i+1] < 0:
            # Linear interpolation to find zero crossing
            x1, x2 = x_values[i], x_values[i+1]
            f1, f2 = diff[i], diff[i+1]
            root = x1 - f1 * (x2 - x1) / (f2 - f1)
            if x_range[0] <= root <= x_range[1]:
                crossovers.append(root)
    # Remove near-duplicates
    unique = []
    for x in crossovers:
        if not any(abs(x - u) < tol for u in unique):
            unique.append(x)
    return unique


def verify_complexity_ceiling(model: AdaptabilityModel,
                               x_range: Tuple[float, float] = (-0.5, 0.5),
                               num_points: int = 2000,
                               tol: float = 1e-6) -> List[float]:
    """Search for configurations where three or more modes share the same decay exponent."""
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    triple_points = []
    n_modes = len(model.n_ord)
    if n_modes < 3:
        return triple_points
    for x in x_values:
        m_vals = [model.M_n(x, n) for n in model.n_ord]
        if any(np.isinf(v) for v in m_vals):
            continue
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                for k in range(j + 1, n_modes):
                    if (abs(m_vals[i] - m_vals[j]) < tol and
                        abs(m_vals[i] - m_vals[k]) < tol):
                        triple_points.append(x)
    # Unique points only
    unique = []
    for x in triple_points:
        if not any(abs(x - u) < tol for u in unique):
            unique.append(x)
    return unique
