# experiments/check_step1_data_integrity.py
"""
Check if Step 1 raw data matches the processed results
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import RAW_DATA_DIR, WARMUP
from utils.io_manager import load_npz, load_pkl


def check_data_integrity(tau=9.0):
    """
    Cross-validate raw spikes against processed avalanche stats
    """
    print("\n" + "="*70)
    print("DATA INTEGRITY CHECK")
    print("="*70)
    print(f"Checking tau_d_I = {tau} ms\n")
    
    # Load raw spikes
    raw_file = Path(RAW_DATA_DIR) / f"spikes_{tau:.1f}.npz"
    if not raw_file.exists():
        print(f"❌ ERROR: Raw data not found: {raw_file}")
        return
    
    raw_data = load_npz(raw_file)
    spike_times = raw_data['spike_times']
    duration = float(raw_data['duration'])
    
    print("RAW DATA:")
    print(f"  Total spikes: {len(spike_times)}")
    print(f"  Duration: {duration:.1f} ms")
    print(f"  Time range: {spike_times.min():.2f} - {spike_times.max():.2f} ms")
    
    # After warmup removal
    spikes_after_warmup = spike_times[spike_times > WARMUP]
    effective_duration = duration - WARMUP
    
    print(f"\nAFTER WARMUP REMOVAL (>{WARMUP} ms):")
    print(f"  Remaining spikes: {len(spikes_after_warmup)}")
    print(f"  Effective duration: {effective_duration:.1f} ms")
    print(f"  Time range: {spikes_after_warmup.min():.2f} - {spikes_after_warmup.max():.2f} ms")
    
    # Calculate actual rates
    if len(spikes_after_warmup) > 0:
        actual_rate = len(spikes_after_warmup) / effective_duration / 800 * 1000
        print(f"  Actual firing rate: {actual_rate:.2f} Hz")
        
        # Calculate actual population ISI
        time_span = spikes_after_warmup.max() - spikes_after_warmup.min()
        actual_pop_isi = time_span / len(spikes_after_warmup)
        print(f"  Actual population ISI: {actual_pop_isi:.5f} ms")
    
    # Load processed stats
    from configs.model_config import PROCESSED_DATA_DIR
    processed_file = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau:.1f}.pkl"
    
    if not processed_file.exists():
        print(f"\n⚠️  Processed file not found: {processed_file}")
        print("   Run: python analysis/avalanche_metrics.py")
        return
    
    processed = load_pkl(processed_file)
    preprocessing = processed['preprocessing']
    
    print(f"\nPROCESSED STATS:")
    print(f"  Mean ISI (reported): {preprocessing['mean_isi']:.5f} ms")
    print(f"  Bin width: {preprocessing['bin_width']:.5f} ms")
    print(f"  Number of bins: {preprocessing['n_bins']}")
    print(f"  Avalanches detected: {preprocessing['n_avalanches']}")
    
    # Check consistency
    print(f"\n" + "="*70)
    print("CONSISTENCY CHECK")
    print("="*70)
    
    if len(spikes_after_warmup) > 0:
        rate_match = abs(actual_rate - 10.0) < 5.0  # Allow 5Hz tolerance
        isi_match = abs(actual_pop_isi - preprocessing['mean_isi']) / preprocessing['mean_isi'] < 0.2
        
        if rate_match:
            print(f"✓ Firing rate reasonable: {actual_rate:.2f} Hz")
        else:
            print(f"⚠️  Firing rate questionable: {actual_rate:.2f} Hz")
        
        if isi_match:
            print(f"✓ ISI consistent: {actual_pop_isi:.5f} vs {preprocessing['mean_isi']:.5f}")
        else:
            print(f"❌ ISI MISMATCH:")
            print(f"   Raw calculation: {actual_pop_isi:.5f} ms")
            print(f"   Processed value: {preprocessing['mean_isi']:.5f} ms")
            print(f"   Difference: {abs(actual_pop_isi - preprocessing['mean_isi']):.5f} ms")
            print(f"   Possible causes:")
            print(f"     1. Binning changed the effective duration")
            print(f"     2. Avalanche detection filtered some spikes")
            print(f"     3. Preprocessing algorithm error")
    
    # Check if avalanche binning makes sense
    n_aval = preprocessing['n_avalanches']
    n_bins = preprocessing['n_bins']
    
    print(f"\nAVALANCHE BINNING SANITY:")
    if n_aval > 0:
        avg_aval_length = n_bins / n_aval
        print(f"  Average avalanche length: {avg_aval_length:.1f} bins")
        
        if avg_aval_length < 2:
            print(f"  ⚠️  Very short avalanches (< 2 bins)")
            print(f"     This suggests bin width might be too small")
        elif avg_aval_length > 100:
            print(f"  ⚠️  Very long avalanches (> 100 bins)")
            print(f"     This suggests bin width might be too large")
            print(f"     Or network lacks quiet periods (supercritical)")
        else:
            print(f"  ✓ Reasonable avalanche lengths")
    
    # Direct comparison with baseline measurement
    print(f"\n" + "="*70)
    print("COMPARISON WITH BASELINE MEASUREMENT")
    print("="*70)
    print(f"From diagnostic script (recent run): ~11.5 Hz")
    print(f"From raw data (stored):              {actual_rate:.2f} Hz")
    print(f"From avalanche ISI:                  {1000/preprocessing['mean_isi']/800:.2f} Hz")
    
    if abs(actual_rate - 11.5) > 2.0:
        print(f"\n⚠️  WARNING: Raw data rate differs from diagnostic")
        print(f"   This suggests network parameters changed between runs")
        print(f"   Or random seed effects are very strong")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=9.0)
    args = parser.parse_args()
    
    check_data_integrity(args.tau)