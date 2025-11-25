# %% Stage 4: Analysis - Spike Train Preprocessing and Avalanche Detection
# Implements Appendix B algorithms with CORRECT population ISI calculation

import numpy as np
from pathlib import Path
from configs.model_config import RAW_DATA_DIR, WARMUP


def load_spike_data(tau_d_I: float):
    """Load raw spike data for a given tau_d_I."""
    filename = f"spikes_{tau_d_I:.1f}.npz"
    filepath = Path(RAW_DATA_DIR) / filename
    
    data = np.load(filepath)
    return data['spike_times'], data['neuron_indices'], float(data['duration'])


def compute_population_ISI(spike_times, warmup_cutoff=None):
    """
    Compute mean inter-spike interval of POPULATION spike train.
    
    ⚠️  CRITICAL FIX (from diagnostics):
    This must be population-level ISI, NOT neuron-level ISI.
    
    Formula: <ISI>_pop = (max_time - min_time) / total_spikes
    
    This gives the average time interval between consecutive spikes
    in the population spike train (regardless of which neuron fires).
    
    Expected range: 0.05 - 0.5 ms for 800 E neurons @ ~5Hz each
    
    Args:
        spike_times: array of spike times
        warmup_cutoff: if provided, exclude spikes before this time (ms)
    
    Returns:
        mean_isi: population ISI in milliseconds
    """
    # Step 1: Remove warm-up transient (as per paper Appendix B)
    if warmup_cutoff is None:
        warmup_cutoff = WARMUP
    
    spike_times_filtered = spike_times[spike_times > warmup_cutoff]
    
    if len(spike_times_filtered) < 10:
        return 0.5  # fallback for empty data
    
    # Step 2: Compute population ISI correctly
    # This is: time_span / number_of_population_spikes
    time_span = spike_times_filtered.max() - spike_times_filtered.min()
    n_spikes = len(spike_times_filtered)
    
    population_isi = time_span / n_spikes
    
    # Sanity check: warn if suspiciously large
    if population_isi > 1.0:
        print(f"    ⚠️  WARNING: Population ISI is {population_isi:.4f} ms (unusually large)")
        print(f"    This suggests very sparse firing or data issues")
    
    return population_isi


def bin_spike_train(spike_times, bin_width, duration, warmup_cutoff=None):
    """
    Bin population spike train with given bin width.
    
    CRITICAL: Remove warm-up period BEFORE binning, to avoid false initial avalanche.
    
    Args:
        spike_times: array of spike times (absolute time coordinates)
        bin_width: width of each bin (ms)
        duration: duration of analysis window (ms, should be DURATION - WARMUP)
        warmup_cutoff: if provided, exclude spikes before this time
    
    Returns:
        spike_counts: array of spike counts per bin
        bin_edges: array of bin edge times (relative to warmup_cutoff)
    """
    # Remove warm-up period
    if warmup_cutoff is None:
        warmup_cutoff = WARMUP
    
    spike_times_filtered = spike_times[spike_times > warmup_cutoff]
    
    # Shift times to start from 0 (relative coordinates)
    spike_times_relative = spike_times_filtered - warmup_cutoff
    
    # Create histogram with NO gaps, NO lower limits on bin_width
    n_bins = int(np.ceil(duration / bin_width))
    spike_counts, bin_edges = np.histogram(
        spike_times_relative, 
        bins=n_bins, 
        range=(0, duration)
    )
    
    return spike_counts, bin_edges


def detect_avalanches(spike_counts):
    """
    Detect avalanches from binned spike train.
    
    Avalanche definition (Appendix B of paper):
    - Starts at first non-empty bin (spikes > 0)
    - Ends at next empty bin (spikes = 0)
    - Size S = total spikes during avalanche
    - Duration T = number of bins spanned
    
    ⚠️  CRITICAL REQUIREMENT:
    This algorithm REQUIRES empty bins to separate avalanches.
    If there are NO empty bins, the entire record is treated as ONE avalanche.
    
    This indicates a fundamental problem:
    - Network is over-excited (too much external input)
    - Or inhibition is too weak
    - Or time binning is wrong
    
    Returns:
        avalanche_sizes: array of avalanche sizes
        avalanche_durations: array of avalanche durations (in bins)
    """
    avalanche_sizes = []
    avalanche_durations = []
    
    in_avalanche = False
    current_size = 0
    current_duration = 0
    
    for count in spike_counts:
        if count > 0:
            if not in_avalanche:
                # Start new avalanche
                in_avalanche = True
                current_size = count
                current_duration = 1
            else:
                # Continue avalanche
                current_size += count
                current_duration += 1
        else:  # count == 0
            if in_avalanche:
                # End avalanche
                avalanche_sizes.append(current_size)
                avalanche_durations.append(current_duration)
                
                in_avalanche = False
                current_size = 0
                current_duration = 0
    
    # Handle case where avalanche extends to end
    if in_avalanche:
        avalanche_sizes.append(current_size)
        avalanche_durations.append(current_duration)
    
    return np.array(avalanche_sizes), np.array(avalanche_durations)


def preprocess_single_tau(tau_d_I: float):
    """
    Complete preprocessing pipeline for a single tau_d_I value.
    
    Implements the correct procedure from Appendix B of the paper:
    1. Remove warm-up transient (first 500ms)
    2. Compute population-level ISI
    3. Bin spike train
    4. Detect avalanches via zero-crossing
    """
    # Load raw data
    spike_times, neuron_indices, duration = load_spike_data(tau_d_I)
    
    print(f"\ntau_d_I={tau_d_I:.1f}ms: Preprocessing...")
    print(f"  Raw data: {len(spike_times)} spikes over {duration:.1f} ms")
    
    # Step 1: Compute POPULATION ISI correctly
    # This is the KEY FIX!
    mean_isi = compute_population_ISI(spike_times, warmup_cutoff=WARMUP)
    print(f"  Population ISI: {mean_isi:.5f} ms")
    
    # Step 2: Use ISI as bin width
    bin_width = mean_isi
    
    # Step 3: Bin the spike train (excluding warm-up)
    spike_counts, bin_edges = bin_spike_train(
        spike_times, 
        bin_width, 
        duration,
        warmup_cutoff=WARMUP
    )
    
    n_empty_bins = np.sum(spike_counts == 0)
    n_total_bins = len(spike_counts)
    
    print(f"  Binning: {n_total_bins} bins of width {bin_width:.5f} ms")
    print(f"  Empty bins: {n_empty_bins} / {n_total_bins} ({100*n_empty_bins/n_total_bins:.1f}%)")
    
    # Step 4: Detect avalanches
    sizes, durations = detect_avalanches(spike_counts)
    
    print(f"  Avalanches detected: {len(sizes)}")
    if len(sizes) > 1:
        print(f"    Size range: {sizes.min()} - {sizes.max()} (mean: {sizes.mean():.1f})")
        print(f"    Duration range: {durations.min()} - {durations.max()} bins")
        print(f"    ✓ Binning strategy is working correctly!")
    elif len(sizes) == 1:
        print(f"    ⚠️  WARNING: Only 1 avalanche detected!")
        print(f"    Possible causes:")
        print(f"      1. Bin width is too large (all spikes grouped)")
        print(f"      2. Network is in persistent/epileptic state (no quiet periods)")
        print(f"      3. External input too strong, or inhibition too weak")
    else:
        print(f"    ❌ ERROR: No avalanches detected! Data or parameters issue.")
    
    # Step 5: Convert durations from bins to milliseconds
    durations_ms = durations * bin_width
    
    stats = {
        'mean_isi': mean_isi,
        'bin_width': bin_width,
        'n_bins': len(spike_counts),
        'n_avalanches': len(sizes),
        'avalanche_sizes': sizes,
        'avalanche_durations': durations_ms
    }
    
    return stats