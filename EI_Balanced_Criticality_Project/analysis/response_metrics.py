import numpy as np

def compute_sensitivity(r_baseline, r_signal):
    """
    Calculate Response Sensitivity.
    Formula: Delta_r / r_baseline
    
    Args:
        r_baseline: Mean firing rate under background input only (Hz).
        r_signal: Mean firing rate under background + signal (Hz).
    """
    if r_baseline == 0:
        return 0.0
    return (r_signal - r_baseline) / r_baseline


def compute_reliability(spike_counts, bin_size_ms, step_size_ms, dt):
    """
    Calculate Response Reliability using Fano Factor.
    Ref: Eq in Prompt & Appendix B.
    
    Args:
        spike_counts: Array of shape (n_trials, n_time_steps). 
                      Total population spike count per step.
        bin_size_ms: Width of the window (e.g., 50ms).
        step_size_ms: Sliding step (e.g., 20ms).
        dt: Simulation step size (ms).
        
    Returns:
        reliability: Scalar value.
    """
    n_trials, n_steps = spike_counts.shape
    
    # Convert window sizes to indices
    bin_width = int(bin_size_ms / dt)
    step_width = int(step_size_ms / dt)
    
    fano_factors = []
    
    # Sliding window
    for start_idx in range(0, n_steps - bin_width, step_width):
        end_idx = start_idx + bin_width
        
        # 1. Count spikes in this bin for all trials
        # Shape: (n_trials,)
        counts_in_bin = np.sum(spike_counts[:, start_idx:end_idx], axis=1)
        
        # 2. Calculate Mean and Variance across trials
        mu = np.mean(counts_in_bin)
        var = np.var(counts_in_bin)
        
        # 3. Calculate Fano Factor (Var / Mean)
        # Avoid division by zero if mean is very small
        if mu > 1e-5:
            ff = var / mu
            fano_factors.append(ff)
    
    if not fano_factors:
        return 0.0
        
    # 4. Reliability = E[1/FF]
    # Note: Avoid 1/0. Paper implies FF is typically > 0 for Poisson-like processes.
    # We add a small epsilon or filter zeros to be safe.
    fano_factors = np.array(fano_factors)
    valid_ff = fano_factors[fano_factors > 1e-5]
    
    if len(valid_ff) == 0:
        return 0.0
        
    reliability = np.mean(1.0 / valid_ff)
    
    return reliability