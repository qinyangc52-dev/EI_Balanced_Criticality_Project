# experiments/validate_step1_results.py
"""
Quick validator for Step 1 results
Tells you if you need to redo Step 1 or can proceed with signal enhancement
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.io_manager import load_pkl
from configs.model_config import PROCESSED_DATA_DIR


def validate_step1(tau_critical=8.5):
    """
    Validate if Step 1 found a true critical point.
    """
    print("\n" + "="*70)
    print("STEP 1 VALIDATION")
    print("="*70)
    print(f"Checking tau_d_I = {tau_critical} ms\n")
    
    # Load avalanche results
    try:
        filepath = Path(PROCESSED_DATA_DIR) / f"avalanche_stats_{tau_critical:.1f}.pkl"
        results = load_pkl(filepath)
    except FileNotFoundError:
        print(f"❌ ERROR: No results found for tau = {tau_critical}")
        print(f"   Run Step 1 first: python main.py --tau {tau_critical}")
        return None
    
    # Extract metrics
    size_exp = results.get('size_exponent')
    dur_exp = results.get('duration_exponent')
    mean_isi = results['preprocessing']['mean_isi']
    n_aval = results['preprocessing']['n_avalanches']
    size_ks = results.get('size_ks')
    
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Size exponent (τ):       {size_exp:.3f}" if size_exp else "  Failed")
    print(f"Duration exponent (α):   {dur_exp:.3f}" if dur_exp else "  Failed")
    print(f"KS distance (size):      {size_ks:.3f}" if size_ks else "  N/A")
    print(f"Number of avalanches:    {n_aval}")
    print(f"Mean ISI:                {mean_isi:.5f} ms")
    
    # Calculate implied firing rate
    # ISI = time_span / n_spikes
    # If ISI = 0.1 ms, then rate ≈ 1/0.0001 s = 10 kHz total
    # For 800 neurons: 10000/800 = 12.5 Hz per neuron
    # This is rough, but gives an idea
    
    if mean_isi > 0:
        implied_rate = 1.0 / (mean_isi / 1000.0) / 800  # Hz per neuron
    else:
        implied_rate = 0
    
    print(f"Implied firing rate:     {implied_rate:.2f} Hz")
    
    print("\n" + "="*70)
    print("QUALITY ASSESSMENT")
    print("="*70)
    
    score = 0
    max_score = 5
    issues = []
    
    # Check 1: Size exponent
    if size_exp is not None:
        if 1.3 <= size_exp <= 2.0:
            print("✓ Size exponent in valid range (1.3-2.0)")
            score += 1
        else:
            print(f"⚠️  Size exponent out of range: {size_exp:.3f}")
            issues.append(f"Size exponent = {size_exp:.3f} (expect 1.3-2.0)")
    else:
        print("❌ Size exponent fitting failed")
        issues.append("Size exponent fitting failed")
    
    # Check 2: KS distance
    if size_ks is not None:
        if size_ks < 0.15:
            print("✓ KS distance good (< 0.15)")
            score += 1
        else:
            print(f"⚠️  KS distance high: {size_ks:.3f}")
            issues.append(f"KS distance = {size_ks:.3f} (expect < 0.15)")
    
    # Check 3: Number of avalanches
    if n_aval > 1000:
        print(f"✓ Sufficient avalanches: {n_aval}")
        score += 1
    elif n_aval > 500:
        print(f"⚠️  Moderate avalanches: {n_aval} (prefer > 1000)")
        score += 0.5
    else:
        print(f"❌ Too few avalanches: {n_aval}")
        issues.append(f"Only {n_aval} avalanches (expect > 1000)")
    
    # Check 4: Mean ISI
    if 0.05 <= mean_isi <= 0.3:
        print(f"✓ Mean ISI in good range: {mean_isi:.5f} ms")
        score += 1
    elif 0.3 < mean_isi <= 0.5:
        print(f"⚠️  Mean ISI slightly high: {mean_isi:.5f} ms")
        score += 0.5
        issues.append(f"ISI = {mean_isi:.5f} ms, implies high firing rate")
    else:
        print(f"❌ Mean ISI problematic: {mean_isi:.5f} ms")
        issues.append(f"ISI = {mean_isi:.5f} ms (expect 0.05-0.3)")
    
    # Check 5: Firing rate
    if 3 <= implied_rate <= 7:
        print(f"✓ Implied rate reasonable: {implied_rate:.2f} Hz")
        score += 1
    elif 7 < implied_rate <= 10:
        print(f"⚠️  Implied rate high: {implied_rate:.2f} Hz")
        score += 0.5
        issues.append(f"Firing rate ≈ {implied_rate:.2f} Hz (prefer 3-7 Hz)")
    else:
        print(f"❌ Implied rate too high: {implied_rate:.2f} Hz")
        issues.append(f"Firing rate ≈ {implied_rate:.2f} Hz (expect 3-7 Hz)")
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"Quality Score: {score:.1f}/{max_score}")
    
    if score >= 4.5:
        print("\n✅ EXCELLENT: Step 1 results are solid")
        print("   Recommendation: Proceed with METHOD B (enhance signal)")
        print("   Action: Use signal_strength=8.0, signal_freq=50.0")
        recommendation = "METHOD_B"
        
    elif score >= 3.0:
        print("\n⚠️  ACCEPTABLE: Step 1 results are usable but not ideal")
        print("   Recommendation: Try METHOD B first")
        print("   If still problematic, consider METHOD C (redo Step 1)")
        recommendation = "METHOD_B_OR_C"
        
    else:
        print("\n❌ PROBLEMATIC: Step 1 results are questionable")
        print("   Recommendation: Use METHOD C (redo Step 1)")
        print("   Reason: Current critical point may be incorrect")
        recommendation = "METHOD_C"
    
    if issues:
        print("\n📋 Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    print("\n" + "="*70)
    print("DETAILED RECOMMENDATIONS")
    print("="*70)
    
    if recommendation == "METHOD_B":
        print("""
METHOD B: ENHANCE SIGNAL (Keep current Step 1)

PROS:
  ✓ No need to rerun Step 1 (saves ~3 hours)
  ✓ Your avalanche analysis is already good
  ✓ Can get results quickly

CONS:
  ⚠️  Reliability might be lower than paper
  ⚠️  Network is in high-activity regime

IMPLEMENTATION:
  1. Edit experiments/run_criticality_metrics.py:
     - Line ~40: signal_freq = 50.0
     - Line ~85 & ~135: signal_strength = 8.0
  
  2. Run: python run_criticality_metrics.py --quick
  
  3. Expect:
     - Sensitivity: 0.10-0.20 (positive!)
     - Reliability: 0.3-0.5 (moderate)
     - Baseline: ~11 Hz (stays high)
        """)
        
    elif recommendation == "METHOD_C":
        print("""
METHOD C: REDO STEP 1 (Recommended for publication)

PROS:
  ✓ Find true critical point
  ✓ Results match paper better
  ✓ Both Step 1 & 2 will be consistent

CONS:
  ⚠️  Need to rerun all simulations (~3 hours)
  ⚠️  More work upfront

IMPLEMENTATION:
  1. Backup current results:
     mkdir data/backup
     cp -r data/processed data/backup/
  
  2. Edit configs/model_config.py:
     EXT_FREQ_TOTAL = 2500.0  # Reduce from 4000
  
  3. Rerun Step 1:
     python main.py --all
  
  4. Validate new critical point:
     python experiments/validate_step1_results.py
  
  5. Then run Step 2:
     python experiments/run_criticality_metrics.py
        """)
        
    else:  # METHOD_B_OR_C
        print("""
METHOD B OR C: Your choice

Try METHOD B first (quick, 10 minutes):
  - If Sensitivity > 0.15 → Good enough!
  - If still negative → Use METHOD C

Use METHOD C if:
  - You need publication-quality results
  - You have time for proper tuning
  - Step 2 results from METHOD B are unsatisfactory
        """)
    
    print("\n" + "="*70 + "\n")
    
    return {
        'score': score,
        'recommendation': recommendation,
        'issues': issues,
        'metrics': {
            'size_exp': size_exp,
            'n_aval': n_aval,
            'mean_isi': mean_isi,
            'implied_rate': implied_rate
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=8.5,
                       help='tau_d_I value to validate')
    args = parser.parse_args()
    
    result = validate_step1(args.tau)