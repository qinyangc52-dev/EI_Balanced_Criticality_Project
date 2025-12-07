# experiments/diagnose_criticality.py
"""
Diagnostic script to identify issues with sensitivity/reliability measurements
"""

import sys
from pathlib import Path
import numpy as np
import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import N_E, DT, SEED, EXT_FREQ_TOTAL, N_EXT, EXT_RATE
from models.network import BalancedNetwork
from core.frozen_input import FrozenSignalInjector, BaselineInput
from experiments.frozen_signal_generator import FrozenSignalGenerator


def diagnose_baseline_activity(tau_d_I=8.5, duration=2000.0):
    """
    Diagnose 1: Check baseline firing rate
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Baseline Network Activity")
    print("="*70)
    print(f"Testing tau_d_I = {tau_d_I} ms")
    
    bm.clear_name_cache()
    bm.random.seed(SEED)
    
    net = BalancedNetwork(tau_d_I=tau_d_I)
    inp = BaselineInput(net)
    
    runner = bp.DSRunner(
        bp.DynSysGroup(net=net, inp=inp),
        monitors={'spikes': net.E.spike, 'V': net.E.V},
        dt=DT,
        progress_bar=True
    )
    
    runner.run(duration)
    
    # Calculate firing rate
    spikes = runner.mon['spikes']
    total_spikes = np.sum(spikes)
    rate = total_spikes / duration / N_E * 1000.0
    
    # Calculate mean voltage
    mean_V = np.mean(runner.mon['V'])
    
    print(f"\nResults:")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Mean firing rate: {rate:.2f} Hz")
    print(f"  Mean membrane potential: {mean_V:.2f} mV")
    print(f"  External input rate: {EXT_RATE:.2f} Hz")
    print(f"  External freq total: {EXT_FREQ_TOTAL:.2f} Hz")
    print(f"  N_ext: {N_EXT}")
    
    print(f"\nDiagnosis:")
    if rate > 8.0:
        print(f"  ❌ PROBLEM: Firing rate too HIGH ({rate:.2f} Hz > 8 Hz)")
        print(f"     → Network is SUPERCRITICAL (over-excited)")
        print(f"     → Reduce EXT_FREQ_TOTAL in model_config.py")
        print(f"     → Current: {EXT_FREQ_TOTAL}, Try: 2000-3000")
    elif rate < 2.0:
        print(f"  ❌ PROBLEM: Firing rate too LOW ({rate:.2f} Hz < 2 Hz)")
        print(f"     → Network is SUBCRITICAL (under-excited)")
        print(f"     → Increase EXT_FREQ_TOTAL in model_config.py")
    else:
        print(f"  ✓ OK: Firing rate in reasonable range ({rate:.2f} Hz)")
    
    return rate, spikes


def diagnose_signal_strength(tau_d_I=8.5, duration=2000.0):
    """
    Diagnose 2: Check signal injection effect
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Signal Injection Strength")
    print("="*70)
    
    # Generate frozen signal
    gen = FrozenSignalGenerator(duration_ms=duration, signal_freq=25.0)
    signal_E = gen.generate_signal('E')
    signal_I = gen.generate_signal('I')
    
    print(f"Signal properties:")
    print(f"  Signal shape: {signal_E.shape}")
    print(f"  Total signal spikes: {np.sum(signal_E)}")
    print(f"  Signal rate per neuron: {np.sum(signal_E)/duration/N_E*1000:.2f} Hz")
    
    # Test different signal strengths
    strengths = [0.5, 1.0, 2.0, 5.0]
    rates = []
    
    print(f"\nTesting signal strengths...")
    
    for strength in strengths:
        bm.clear_name_cache()
        bm.random.seed(SEED)
        
        net = BalancedNetwork(tau_d_I=tau_d_I)
        inp = FrozenSignalInjector(net, signal_E, signal_I, signal_strength=strength)
        
        runner = bp.DSRunner(
            bp.DynSysGroup(net=net, inp=inp),
            monitors={'spikes': net.E.spike},
            dt=DT,
            progress_bar=False
        )
        
        runner.run(duration)
        
        rate = np.mean(runner.mon['spikes']) * 1000.0 / DT
        rates.append(rate)
        print(f"  Strength {strength:4.1f}x: Rate = {rate:.2f} Hz")
    
    print(f"\nDiagnosis:")
    rate_increase = [(rates[i] - rates[0])/rates[0]*100 for i in range(len(rates))]
    
    if rate_increase[-1] < 10:
        print(f"  ❌ PROBLEM: Signal has weak effect (max increase: {rate_increase[-1]:.1f}%)")
        print(f"     → Signal is too weak or drowned by background")
        print(f"     → Try using signal_strength = 5.0 or higher")
        print(f"     → Or increase signal_freq from 25 to 50 Hz")
    else:
        print(f"  ✓ OK: Signal has noticeable effect (max increase: {rate_increase[-1]:.1f}%)")
        best_strength = strengths[np.argmax([r-rates[0] for r in rates])]
        print(f"     → Recommended signal_strength: {best_strength}")
    
    return strengths, rates


def diagnose_background_vs_signal_balance(tau_d_I=8.5, duration=1000.0):
    """
    Diagnose 3: Check background noise vs signal balance
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Background vs Signal Balance")
    print("="*70)
    
    # Generate signal
    gen = FrozenSignalGenerator(duration_ms=duration, signal_freq=25.0)
    signal_E = gen.generate_signal('E')
    signal_I = gen.generate_signal('I')
    
    # Test 1: Pure background
    bm.clear_name_cache()
    bm.random.seed(SEED)
    net1 = BalancedNetwork(tau_d_I=tau_d_I)
    inp1 = BaselineInput(net1)
    runner1 = bp.DSRunner(
        bp.DynSysGroup(net=net1, inp=inp1),
        monitors={'spikes': net1.E.spike},
        dt=DT,
        progress_bar=False
    )
    runner1.run(duration)
    rate_bg = np.mean(runner1.mon['spikes']) * 1000.0 / DT
    
    # Test 2: Background + Signal (strength=1.0)
    bm.clear_name_cache()
    bm.random.seed(SEED)
    net2 = BalancedNetwork(tau_d_I=tau_d_I)
    inp2 = FrozenSignalInjector(net2, signal_E, signal_I, signal_strength=1.0)
    runner2 = bp.DSRunner(
        bp.DynSysGroup(net=net2, inp=inp2),
        monitors={'spikes': net2.E.spike},
        dt=DT,
        progress_bar=False
    )
    runner2.run(duration)
    rate_sig = np.mean(runner2.mon['spikes']) * 1000.0 / DT
    
    # Test 3: Background + Strong Signal (strength=3.0)
    bm.clear_name_cache()
    bm.random.seed(SEED + 1000)
    net3 = BalancedNetwork(tau_d_I=tau_d_I)
    inp3 = FrozenSignalInjector(net3, signal_E, signal_I, signal_strength=3.0)
    runner3 = bp.DSRunner(
        bp.DynSysGroup(net=net3, inp=inp3),
        monitors={'spikes': net3.E.spike},
        dt=DT,
        progress_bar=False
    )
    runner3.run(duration)
    rate_sig_strong = np.mean(runner3.mon['spikes']) * 1000.0 / DT
    
    print(f"\nResults:")
    print(f"  Background only:          {rate_bg:.2f} Hz")
    print(f"  Background + Signal (1x): {rate_sig:.2f} Hz  (Δ = {rate_sig-rate_bg:+.2f} Hz)")
    print(f"  Background + Signal (3x): {rate_sig_strong:.2f} Hz  (Δ = {rate_sig_strong-rate_bg:+.2f} Hz)")
    
    print(f"\nDiagnosis:")
    if rate_sig < rate_bg:
        print(f"  ❌ CRITICAL: Signal DECREASES firing rate!")
        print(f"     → This is the main problem causing negative sensitivity")
        print(f"     → Possible causes:")
        print(f"       1. Signal is being injected as inhibitory instead of excitatory")
        print(f"       2. Background is too strong, overwhelming the signal")
        print(f"       3. Network is over-inhibited when signal arrives")
        print(f"     → Solutions:")
        print(f"       A. Increase signal_strength to 3.0-5.0")
        print(f"       B. Reduce EXT_FREQ_TOTAL to lower baseline")
        print(f"       C. Check FrozenSignalInjector implementation")
    elif (rate_sig - rate_bg) / rate_bg < 0.1:
        print(f"  ⚠️  WARNING: Signal effect is weak (< 10% increase)")
        print(f"     → Increase signal_strength or signal_freq")
    else:
        print(f"  ✓ OK: Signal increases firing rate appropriately")


def plot_diagnostic_raster(tau_d_I=8.5, duration=1000.0):
    """
    Visual diagnostic: Plot spike rasters
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Visual Spike Raster")
    print("="*70)
    
    gen = FrozenSignalGenerator(duration_ms=duration, signal_freq=25.0)
    signal_E = gen.generate_signal('E')
    signal_I = gen.generate_signal('I')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Baseline
    bm.clear_name_cache()
    bm.random.seed(SEED)
    net1 = BalancedNetwork(tau_d_I=tau_d_I)
    inp1 = BaselineInput(net1)
    runner1 = bp.DSRunner(
        bp.DynSysGroup(net=net1, inp=inp1),
        monitors={'spikes': net1.E.spike},
        dt=DT,
        progress_bar=False
    )
    runner1.run(duration)
    
    spikes1 = runner1.mon['spikes']
    spike_times1, spike_neurons1 = np.where(spikes1)
    spike_times1 = spike_times1 * DT
    
    axes[0].scatter(spike_times1, spike_neurons1, s=1, c='black', alpha=0.5)
    axes[0].set_ylabel('Neuron ID', fontsize=12)
    axes[0].set_title('Baseline (Background Only)', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, duration)
    axes[0].set_ylim(0, 100)  # Show first 100 neurons
    
    # With Signal
    bm.clear_name_cache()
    bm.random.seed(SEED)
    net2 = BalancedNetwork(tau_d_I=tau_d_I)
    inp2 = FrozenSignalInjector(net2, signal_E, signal_I, signal_strength=3.0)
    runner2 = bp.DSRunner(
        bp.DynSysGroup(net=net2, inp=inp2),
        monitors={'spikes': net2.E.spike},
        dt=DT,
        progress_bar=False
    )
    runner2.run(duration)
    
    spikes2 = runner2.mon['spikes']
    spike_times2, spike_neurons2 = np.where(spikes2)
    spike_times2 = spike_times2 * DT
    
    axes[1].scatter(spike_times2, spike_neurons2, s=1, c='red', alpha=0.5)
    axes[1].set_xlabel('Time (ms)', fontsize=12)
    axes[1].set_ylabel('Neuron ID', fontsize=12)
    axes[1].set_title('With Signal (3x Strength)', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, duration)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    
    from configs.model_config import FIGURE_DIR
    save_path = Path(FIGURE_DIR) / 'diagnostic_raster.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Raster plot saved to: {save_path}")
    plt.show()


def run_full_diagnostics():
    """Run all diagnostic tests"""
    print("\n" + "🔧"*35)
    print("CRITICALITY METRICS DIAGNOSTIC SUITE")
    print("🔧"*35)
    
    tau = 8.5  # Your critical point
    
    # Run diagnostics
    diagnose_baseline_activity(tau)
    diagnose_signal_strength(tau)
    diagnose_background_vs_signal_balance(tau)
    plot_diagnostic_raster(tau)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the diagnostics above")
    print("2. Apply suggested fixes to model_config.py")
    print("3. Re-run: python run_criticality_metrics.py --quick")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_full_diagnostics()