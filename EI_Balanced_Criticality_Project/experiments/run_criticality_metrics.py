# experiments/run_criticality_metrics.py
"""
Criticality Metrics Experiment: Sensitivity & Reliability
Focus on critical region (8-9ms) with optimized pipeline
"""

import sys
from pathlib import Path
import numpy as np
import brainpy as bp
import brainpy.math as bm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    N_E, DT, SEED, PROCESSED_DATA_DIR
)
from models.network import BalancedNetwork
from core.frozen_input import FrozenSignalInjector, BaselineInput
from experiments.frozen_signal_generator import FrozenSignalGenerator
from analysis.response_metrics import compute_sensitivity, compute_reliability
from utils.io_manager import save_pkl


class CriticalityMetricsExperiment:
    """
    Streamlined experiment to measure Sensitivity and Reliability
    at critical point.
    """
    
    def __init__(self, tau_list=None, duration_ms=2000.0):
        """
        Args:
            tau_list: List of tau_d_I values to test (default: around critical)
            duration_ms: Duration of each trial
        """
        if tau_list is None:
            # Focus on critical region (8-9ms based on your findings)
            self.tau_list = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
        else:
            self.tau_list = tau_list
        
        self.duration = duration_ms
        self.steps = int(duration_ms / DT)
        
        # Generate frozen signal once (reused across all experiments)
        print("Generating frozen signal template...")
        gen = FrozenSignalGenerator(duration_ms=duration_ms, signal_freq=25.0)
        signals = gen.save_signals('criticality_frozen_signal.pkl')
        
        self.frozen_signal_E = signals['signal_E']
        self.frozen_signal_I = signals['signal_I']
        
        print(f"  ✓ Signal shape: {self.frozen_signal_E.shape}")
        print(f"  ✓ Total signal spikes: {np.sum(self.frozen_signal_E)}\n")
    
    def measure_sensitivity(self, tau_d_I, n_trials=20):
        """
        Measure sensitivity: response difference between baseline and signal.
        
        Formula: Sensitivity = (r_signal - r_baseline) / r_baseline
        
        Args:
            tau_d_I: Inhibitory decay time constant
            n_trials: Number of trials to average (reduces noise)
        """
        print(f"  Measuring Sensitivity ({n_trials} trials)...")
        
        r_baseline_trials = []
        r_signal_trials = []
        
        # Run multiple trials with different background noise
        for trial in range(n_trials):
            bm.clear_name_cache()
            bm.random.seed(SEED + trial)
            
            # Baseline trial (background only)
            net_base = BalancedNetwork(tau_d_I=tau_d_I)
            inp_base = BaselineInput(net_base)
            
            runner_base = bp.DSRunner(
                bp.DynSysGroup(net=net_base, inp=inp_base),
                monitors={'spikes': net_base.E.spike},
                dt=DT,
                progress_bar=False
            )
            runner_base.run(self.duration)
            
            # Calculate firing rate (Hz)
            r_baseline = np.mean(runner_base.mon['spikes']) * 1000.0 / DT
            r_baseline_trials.append(r_baseline)
            
            # Signal trial (background + frozen signal)
            bm.clear_name_cache()
            bm.random.seed(SEED + trial + 10000)
            
            net_sig = BalancedNetwork(tau_d_I=tau_d_I)
            inp_sig = FrozenSignalInjector(net_sig, self.frozen_signal_E, self.frozen_signal_I)
            
            runner_sig = bp.DSRunner(
                bp.DynSysGroup(net=net_sig, inp=inp_sig),
                monitors={'spikes': net_sig.E.spike},
                dt=DT,
                progress_bar=False
            )
            runner_sig.run(self.duration)
            
            r_signal = np.mean(runner_sig.mon['spikes']) * 1000.0 / DT
            r_signal_trials.append(r_signal)
        
        # Average across trials
        r_baseline_avg = np.mean(r_baseline_trials)
        r_signal_avg = np.mean(r_signal_trials)
        
        sensitivity = compute_sensitivity(r_baseline_avg, r_signal_avg)
        
        print(f"    Baseline: {r_baseline_avg:.2f} Hz")
        print(f"    Signal:   {r_signal_avg:.2f} Hz")
        print(f"    Sensitivity: {sensitivity:.4f}")
        
        return {
            'sensitivity': sensitivity,
            'r_baseline': r_baseline_avg,
            'r_signal': r_signal_avg,
            'r_baseline_std': np.std(r_baseline_trials),
            'r_signal_std': np.std(r_signal_trials)
        }
    
    def measure_reliability(self, tau_d_I, n_trials=50):
        """
        Measure reliability: response consistency across trials with same signal.
        
        Method: Fano Factor - variance/mean of spike counts across trials.
        Reliability = E[1 / FanoFactor]
        
        Args:
            tau_d_I: Inhibitory decay time constant
            n_trials: Number of trials (more = better statistics)
        """
        print(f"  Measuring Reliability ({n_trials} trials)...")
        
        all_spike_counts = []
        
        # Run multiple trials with SAME frozen signal, different background
        for trial in range(n_trials):
            bm.clear_name_cache()
            bm.random.seed(SEED + 20000 + trial)
            
            net = BalancedNetwork(tau_d_I=tau_d_I)
            inp = FrozenSignalInjector(net, self.frozen_signal_E, self.frozen_signal_I)
            
            runner = bp.DSRunner(
                bp.DynSysGroup(net=net, inp=inp),
                monitors={'spikes': net.E.spike},
                dt=DT,
                progress_bar=False
            )
            runner.run(self.duration)
            
            # Population spike count per time step
            pop_spikes = np.sum(runner.mon['spikes'], axis=1)
            all_spike_counts.append(pop_spikes)
        
        # Shape: (n_trials, time_steps)
        spike_counts_array = np.array(all_spike_counts)
        
        reliability = compute_reliability(
            spike_counts_array,
            bin_size_ms=50.0,
            step_size_ms=20.0,
            dt=DT
        )
        
        print(f"    Reliability: {reliability:.4f}")
        
        return {
            'reliability': reliability,
            'spike_counts': spike_counts_array
        }
    
    def run_full_scan(self, n_sens_trials=20, n_rel_trials=50):
        """
        Run complete scan across tau values.
        
        Args:
            n_sens_trials: Trials for sensitivity measurement
            n_rel_trials: Trials for reliability measurement
        """
        print("\n" + "="*70)
        print("CRITICALITY METRICS EXPERIMENT")
        print("="*70)
        print(f"Testing {len(self.tau_list)} tau_d_I values")
        print(f"Sensitivity trials: {n_sens_trials}")
        print(f"Reliability trials: {n_rel_trials}")
        print("="*70 + "\n")
        
        results = {
            'tau': [],
            'sensitivity': [],
            'reliability': [],
            'r_baseline': [],
            'r_signal': [],
            'details': {}
        }
        
        for tau in self.tau_list:
            print(f"\n{'─'*70}")
            print(f"tau_d_I = {tau:.1f} ms")
            print(f"{'─'*70}")
            
            # Sensitivity
            sens_data = self.measure_sensitivity(tau, n_sens_trials)
            
            # Reliability
            rel_data = self.measure_reliability(tau, n_rel_trials)
            
            # Store results
            results['tau'].append(tau)
            results['sensitivity'].append(sens_data['sensitivity'])
            results['reliability'].append(rel_data['reliability'])
            results['r_baseline'].append(sens_data['r_baseline'])
            results['r_signal'].append(sens_data['r_signal'])
            results['details'][tau] = {
                'sensitivity': sens_data,
                'reliability': rel_data
            }
            
            print(f"  ✓ Complete\n")
        
        # Save results
        save_path = Path(PROCESSED_DATA_DIR) / 'criticality_metrics_results.pkl'
        save_pkl(save_path, results)
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(f"Results saved to: {save_path}")
        
        # Print summary
        print("\nSUMMARY:")
        print(f"{'tau_d_I':>8} {'Sensitivity':>12} {'Reliability':>12} {'r_base':>10} {'r_sig':>10}")
        print("-"*60)
        for i, tau in enumerate(results['tau']):
            print(f"{tau:>8.1f} {results['sensitivity'][i]:>12.4f} "
                  f"{results['reliability'][i]:>12.4f} "
                  f"{results['r_baseline'][i]:>10.2f} {results['r_signal'][i]:>10.2f}")
        
        # Identify critical point
        max_product_idx = np.argmax(
            np.array(results['sensitivity']) * np.array(results['reliability'])
        )
        critical_tau = results['tau'][max_product_idx]
        
        print(f"\n🎯 Critical point (max Sens×Rel): tau_d_I = {critical_tau:.1f} ms")
        print("="*70 + "\n")
        
        return results


def quick_test():
    """Quick test with minimal trials."""
    print("Running QUICK TEST (2 tau values, few trials)...\n")
    
    exp = CriticalityMetricsExperiment(
        tau_list=[7.5, 8.5],  # Just 2 values
        duration_ms=1000.0     # Shorter duration
    )
    
    results = exp.run_full_scan(
        n_sens_trials=5,   # Minimal trials
        n_rel_trials=10
    )
    
    return results


def full_experiment():
    """Full experiment with proper statistics."""
    exp = CriticalityMetricsExperiment(
        tau_list=[7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        duration_ms=2000.0
    )
    
    results = exp.run_full_scan(
        n_sens_trials=20,   # Good statistics
        n_rel_trials=50
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    args = parser.parse_args()
    
    if args.quick:
        results = quick_test()
    else:
        results = full_experiment()