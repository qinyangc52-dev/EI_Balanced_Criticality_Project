# %% Stage 4: Analysis - Avalanche Statistical Metrics
# Computes distributions and criticality measures using MLE (Rigorous Method)

import numpy as np
import powerlaw  # 需要 pip install powerlaw
from pathlib import Path
from configs.model_config import PROCESSED_DATA_DIR
from analysis.preprocessing import preprocess_single_tau
from utils.io_manager import save_pkl


def compute_avalanche_distribution(sizes, durations):
    """
    Compute probability distributions of avalanche size and duration.
    For visualization purposes.
    """
    # Size distribution
    size_counts = np.bincount(sizes.astype(int))
    size_values = np.arange(len(size_counts))
    # Avoid division by zero
    total_size = np.sum(size_counts)
    size_probs = size_counts / total_size if total_size > 0 else size_counts
    
    # Duration distribution
    duration_counts = np.bincount(durations.astype(int))
    duration_values = np.arange(len(duration_counts))
    total_dur = np.sum(duration_counts)
    duration_probs = duration_counts / total_dur if total_dur > 0 else duration_counts
    
    return {
        'size_values': size_values,
        'size_probs': size_probs,
        'duration_values': duration_values,
        'duration_probs': duration_probs
    }


def fit_powerlaw_mle(data, name="Data"):
    """
    Rigorous Power-law fitting using Maximum Likelihood Estimation (MLE).
    Uses the 'powerlaw' library to automatically find optimal x_min.
    
    Args:
        data: Array of integers (sizes or durations)
        name: String for logging
        
    Returns:
        exponent (alpha), KS_distance, x_min
    """
    # 移除 0 或负数，因为幂律定义在 x > 0
    data = data[data > 0]
    
    if len(data) < 50:
        print(f"  [Warning] Not enough data for MLE fit for {name} (n={len(data)})")
        return None, None, None

    try:
        # discrete=True 表示数据是整数（粒子数或时间步数）
        # xmin=None 让库自动寻找最小化 KS 距离的最佳截断点
        results = powerlaw.Fit(data, discrete=True, verbose=False)
        
        alpha = results.power_law.alpha
        ks = results.power_law.KS()
        xmin = results.xmin
        
        return alpha, ks, xmin
        
    except Exception as e:
        print(f"  [Error] MLE fit failed for {name}: {e}")
        return None, None, None


def compute_crackle_noise_relation(sizes, durations):
    """
    Compute crackle noise relation scaling exponent: <S>(T) ~ T^gamma
    We simply use log-log linear regression here as it's a scaling relation, 
    not a probability distribution.
    """
    unique_durations = np.unique(durations)
    
    if len(unique_durations) < 3:
        return None
    
    mean_sizes = []
    valid_durations = []
    
    for dur in unique_durations:
        mask = durations == dur
        if np.sum(mask) >= 5:  # Require at least 5 samples per duration bin
            mean_sizes.append(np.mean(sizes[mask]))
            valid_durations.append(dur)
    
    if len(valid_durations) < 3:
        return None
    
    # Log-log fit
    log_dur = np.log10(valid_durations)
    log_size = np.log10(mean_sizes)
    
    coeffs = np.polyfit(log_dur, log_size, 1)
    slope = coeffs[0]
    
    # slope represents 1 / (sigma * nu * z)
    return slope


def analyze_single_tau(tau_d_I: float):
    """
    Complete analysis pipeline for a single tau_d_I value.
    """
    print(f"\nAnalyzing tau_d_I = {tau_d_I:.1f} ms...")
    
    # 1. Preprocessing (Load raw spikes and bin them)
    stats = preprocess_single_tau(tau_d_I)
    sizes = stats['avalanche_sizes']
    durations = stats['avalanche_durations']
    
    # 2. Compute Distributions (for plotting)
    dist = compute_avalanche_distribution(sizes, durations)
    
    # 3. Rigorous Power-law Fitting (MLE)
    # The paper's Size Exponent is 'tau', Duration Exponent is 'alpha'
    size_tau, size_ks, size_xmin = fit_powerlaw_mle(sizes, "Size")
    dur_alpha, dur_ks, dur_xmin = fit_powerlaw_mle(durations, "Duration")
    
    # 4. Crackle Noise Scaling
    scaling_gamma = compute_crackle_noise_relation(sizes, durations)
    
    # 5. Check Crackling Noise Relation Validity
    # Relation: (alpha - 1) / (tau - 1) approx= gamma
    if size_tau and dur_alpha and scaling_gamma:
        predicted_gamma = (dur_alpha - 1) / (size_tau - 1)
        scaling_error = abs(predicted_gamma - scaling_gamma)
    else:
        predicted_gamma = None
        scaling_error = None

    # Combine results
    results = {
        'tau_d_I': tau_d_I,
        'preprocessing': stats,
        'distribution': dist,
        'size_exponent': size_tau,   # tau
        'size_ks': size_ks,
        'size_xmin': size_xmin,
        'duration_exponent': dur_alpha, # alpha
        'duration_ks': dur_ks,
        'duration_xmin': dur_xmin,
        'scaling_gamma': scaling_gamma, # 1/svz
        'predicted_gamma': predicted_gamma,
        'scaling_error': scaling_error
    }
    
    # Print readable report
    print(f"  [Result] Size Exponent (tau):   {size_tau:.3f} (KS={size_ks:.3f}, xmin={size_xmin})" if size_tau else "  [Result] Size: Failed")
    print(f"  [Result] Dur. Exponent (alpha): {dur_alpha:.3f} (KS={dur_ks:.3f}, xmin={dur_xmin})" if dur_alpha else "  [Result] Dur.: Failed")
    
    if scaling_error is not None:
        print(f"  [Check] Crackling Relation: LHS={scaling_gamma:.3f} vs RHS={predicted_gamma:.3f} (Err={scaling_error:.3f})")
    
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