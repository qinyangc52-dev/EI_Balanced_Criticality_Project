# %% Stage 4: Analysis - Avalanche Statistical Metrics
# Computes distributions and criticality measures

import numpy as np
from pathlib import Path
from configs.model_config import PROCESSED_DATA_DIR, POWERLAW_FIT_MIN_SIZE
from analysis.preprocessing import preprocess_single_tau
from utils.io_manager import save_pkl


def compute_avalanche_distribution(sizes, durations):
    """
    Compute probability distributions of avalanche size and duration.
    """
    # Size distribution
    size_counts = np.bincount(sizes.astype(int))
    size_values = np.arange(len(size_counts))
    size_probs = size_counts / np.sum(size_counts)
    
    # Duration distribution
    duration_counts = np.bincount(durations.astype(int))
    duration_values = np.arange(len(duration_counts))
    duration_probs = duration_counts / np.sum(duration_counts)
    
    return {
        'size_values': size_values,
        'size_probs': size_probs,
        'duration_values': duration_values,
        'duration_probs': duration_probs
    }


def fit_powerlaw_simple(data, x_min=None):
    """
    Simple power-law fitting using linear regression in log-log space.
    For more rigorous fitting, use powerlaw package (left for future).
    
    Returns exponent and goodness-of-fit metric.
    """
    if x_min is None:
        x_min = POWERLAW_FIT_MIN_SIZE
    
    # Filter data >= x_min
    data_filtered = data[data >= x_min]
    
    if len(data_filtered) < 10:
        return None, None
    
    # Compute histogram
    counts, bins = np.histogram(data_filtered, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Remove zeros
    mask = counts > 0
    x = bin_centers[mask]
    y = counts[mask]
    
    if len(x) < 5:
        return None, None
    
    # Log-log linear fit
    log_x = np.log10(x)
    log_y = np.log10(y)
    
    coeffs = np.polyfit(log_x, log_y, 1)
    exponent = -coeffs[0]  # negative slope is the exponent
    
    # R-squared
    y_pred = 10 ** np.polyval(coeffs, log_x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return exponent, r_squared


def compute_crackle_noise_relation(sizes, durations):
    """
    Compute crackle noise relation: 1/sigma_vz ~ (alpha - 1) / (tau - 1)
    
    Where sigma_vz is the scaling exponent: <S>(T) ~ T^(1/sigma_vz)
    This is measured empirically from the data.
    """
    # Group sizes by duration
    unique_durations = np.unique(durations)
    
    if len(unique_durations) < 3:
        return None
    
    mean_sizes = []
    valid_durations = []
    
    for dur in unique_durations:
        mask = durations == dur
        if np.sum(mask) >= 3:  # need sufficient samples
            mean_sizes.append(np.mean(sizes[mask]))
            valid_durations.append(dur)
    
    if len(valid_durations) < 3:
        return None
    
    # Log-log fit: log(<S>) = (1/sigma_vz) * log(T)
    log_dur = np.log10(valid_durations)
    log_size = np.log10(mean_sizes)
    
    coeffs = np.polyfit(log_dur, log_size, 1)
    sigma_vz_inv = coeffs[0]
    
    return sigma_vz_inv


def analyze_single_tau(tau_d_I: float):
    """
    Complete analysis pipeline for a single tau_d_I value.
    """
    print(f"\nAnalyzing tau_d_I = {tau_d_I:.1f} ms...")
    
    # Preprocessing
    stats = preprocess_single_tau(tau_d_I)
    
    sizes = stats['avalanche_sizes']
    durations = stats['avalanche_durations']
    
    # Distributions
    dist = compute_avalanche_distribution(sizes, durations)
    
    # Power-law fitting
    size_exponent, size_r2 = fit_powerlaw_simple(sizes)
    duration_exponent, duration_r2 = fit_powerlaw_simple(durations)
    
    # Crackle noise relation
    sigma_vz_inv = compute_crackle_noise_relation(sizes, durations)
    
    # Combine results
    results = {
        'tau_d_I': tau_d_I,
        'preprocessing': stats,
        'distribution': dist,
        'size_exponent': size_exponent,
        'size_r2': size_r2,
        'duration_exponent': duration_exponent,
        'duration_r2': duration_r2,
        'sigma_vz_inv': sigma_vz_inv
    }
    
    print(f"  Size exponent: {size_exponent:.3f} (R2={size_r2:.3f})" if size_exponent else "  Size: no fit")
    print(f"  Duration exponent: {duration_exponent:.3f} (R2={duration_r2:.3f})" if duration_exponent else "  Duration: no fit")
    
    # Save processed results
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    filename = f"avalanche_stats_{tau_d_I:.1f}.pkl"
    filepath = Path(PROCESSED_DATA_DIR) / filename
    save_pkl(filepath, results)
    
    print(f"  Saved to {filepath}")
    
    return results


def analyze_all_tau(tau_d_I_list):
    """
    Analyze all tau_d_I values in the list.
    """
    all_results = {}
    
    for tau_d_I in tau_d_I_list:
        results = analyze_single_tau(tau_d_I)
        all_results[tau_d_I] = results
    
    return all_results
