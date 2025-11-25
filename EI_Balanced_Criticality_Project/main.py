# %% Main Entry Point
# Run complete pipeline: simulation -> analysis -> (visualization)

import sys
import argparse
from pathlib import Path
from configs.model_config import TAU_DECAY_I_LIST, TAU_DECAY_I_CRITICAL
from experiments.phase_transition_runner import PhaseTransitionRunner
from analysis.avalanche_metrics import analyze_all_tau


def main(tau_d_I=None, run_all=False, analyze_only=False):
    """
    Main execution function.
    
    Args:
        tau_d_I: Single tau_d_I value to simulate (float)
        run_all: If True, simulate all tau_d_I values in config
        analyze_only: If True, skip simulation and only run analysis
    """
    print("\n" + "="*70)
    print("E-I BALANCED NETWORK CRITICALITY PROJECT")
    print("Replication of Yang et al. (2025) PRL")
    print("="*70 + "\n")
    
    # Determine which tau_d_I values to process
    if run_all:
        tau_list = TAU_DECAY_I_LIST
        print(f"Mode: Scanning all tau_d_I values")
        print(f"Range: {tau_list}")
    elif tau_d_I is not None:
        tau_list = [tau_d_I]
        print(f"Mode: Single tau_d_I value = {tau_d_I:.1f} ms")
    else:
        tau_list = [TAU_DECAY_I_CRITICAL]
        print(f"Mode: Default (critical state only)")
        print(f"tau_d_I = {TAU_DECAY_I_CRITICAL:.1f} ms")
    
    print()
    
    # Stage 1: Simulation
    if not analyze_only:
        print("STAGE 1: SIMULATION")
        print("-" * 70)
        
        runner = PhaseTransitionRunner(tau_d_I_list=tau_list)
        sim_results = runner.run_all()
        
        print("\nSimulation stage completed successfully.\n")
    else:
        print("STAGE 1: SIMULATION (SKIPPED)\n")
    
    # Stage 2: Analysis
    print("STAGE 2: ANALYSIS")
    print("-" * 70)
    
    analysis_results = analyze_all_tau(tau_list)
    
    print("\nAnalysis stage completed successfully.\n")
    
    # Summary
    print("="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(tau_list)} tau_d_I value(s):")
    for tau in tau_list:
        results = analysis_results[tau]
        n_aval = results['preprocessing']['n_avalanches']
        exp_size = results['size_exponent']
        exp_dur = results['duration_exponent']
        
        print(f"\n  tau_d_I = {tau:.1f} ms:")
        print(f"    Avalanches: {n_aval}")
        if exp_size:
            print(f"    Size exponent: {exp_size:.3f} (R²={results['size_r2']:.3f})")
        if exp_dur:
            print(f"    Duration exponent: {exp_dur:.3f} (R²={results['duration_r2']:.3f})")
    
    print("\n" + "="*70)
    print("Data saved to:")
    print(f"  Raw spikes: data/raw/")
    print(f"  Processed stats: data/processed/")
    print("\nTo visualize results, use visualization/plot_criticality.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run E-I balanced network criticality simulation and analysis"
    )
    
    parser.add_argument(
        '--tau', 
        type=float, 
        default=None,
        help='Single tau_d_I value to simulate (ms)'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Simulate all tau_d_I values in config'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Skip simulation, only run analysis on existing data'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tau is not None and args.all:
        print("ERROR: Cannot specify both --tau and --all")
        sys.exit(1)
    
    main(tau_d_I=args.tau, run_all=args.all, analyze_only=args.analyze_only)