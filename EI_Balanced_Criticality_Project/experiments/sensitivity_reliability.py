import sys
import numpy as np
import brainpy as bp
import brainpy.math as bm
from pathlib import Path

# Increase recursion depth for deep JAX graphs or complex calls
sys.setrecursionlimit(10000)

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    N_E, N_I, N_EXT, DT, 
    TAU_DECAY_I_LIST, TAU_DECAY_I_CRITICAL,
    PROCESSED_DATA_DIR, SEED
)
from models.network import BalancedNetwork
from models.branching import BranchingProcess
from core.inputs import PoissonInput
from analysis.response_metrics import compute_sensitivity, compute_reliability
from utils.io_manager import save_pkl

# ==========================================
# Helper: Frozen Input Generator
# ==========================================
class FrozenPoissonInjector(bp.DynamicalSystem):
    """
    Injects a pre-calculated (Frozen) Poisson signal + Background Noise.
    """
    def __init__(self, target, frozen_spikes, bg_rate=25.0, g_ext=0.022):
        super().__init__()
        self.target = target
        self.frozen_spikes = bm.asarray(frozen_spikes) # Shape: (T, N)
        self.bg_input = PoissonInput(target.network)   # Reuse existing background class
        # Override background rate if needed, or rely on config defaults
        
        # We need to manually inject the frozen current
        # Assuming simple current injection for the signal part for simplicity, 
        # or mapping it to conductance. Here we map to conductance input.
        self.g_ext = g_ext

    def update(self):
        # 1. Update Background (Random every time)
        self.bg_input.update()
        
        # 2. Add Frozen Signal
        t_idx = bp.share['i']
        # Safe indexing
        idx = bm.minimum(t_idx, self.frozen_spikes.shape[0]-1)
        signal_spike = self.frozen_spikes[idx]
        
        # Inject into Excitatory neurons (simplified as current/conductance step)
        # Note: In a rigorous replication, this should go through synaptic dynamics (DualExp)
        # Here we add it directly to input for efficiency, scaled by weight
        self.target.E.input.value += signal_spike * self.g_ext

# ==========================================
# 1. Balanced Network Trial
# ==========================================
def run_balanced_network_experiment(tau_list=None):
    if tau_list is None:
        tau_list = TAU_DECAY_I_LIST

    print(f"\n{'='*60}")
    print(f"Running Balanced Network: Sensitivity & Reliability")
    print(f"{'='*60}")

    results = {
        'tau': [],
        'sensitivity': [],
        'reliability': []
    }

    # Signal Parameters
    duration = 1000.0 # ms
    steps = int(duration / DT)
    signal_freq = 25.0 # Hz
    n_trials = 100
    
    # Pre-generate ONE frozen signal pattern for all trials
    print("Generating Frozen Signal Template...")
    np.random.seed(SEED)
    prob = signal_freq * DT / 1000.0
    # Shape: (Steps, N_E). Only drive E neurons with signal for clarity
    frozen_signal = (np.random.rand(steps, N_E) < prob).astype(float)

    for tau in tau_list:
        print(f"\nProcessing tau_I = {tau} ms...")
        
        # --- Part A: Sensitivity (Baseline vs Signal) ---
        # 1. Baseline (Background Only)
        bp.math.clear_name_cache() # Clear cache before creating new network
        bm.random.seed(SEED) # Reset seed for consistency
        net_base = BalancedNetwork(tau_d_I=tau)
        input_base = PoissonInput(net_base) # Standard background
        runner_base = bp.DSRunner(
            bp.DynSysGroup(net=net_base, inp=input_base),
            monitors={'spikes': net_base.E.spike},
            dt=DT, progress_bar=False
        )
        runner_base.run(duration)
        r_base = np.mean(runner_base.mon['spikes']) * 1000.0 / DT # Hz per neuron
        
        # 2. Signal (Background + Signal)
        bp.math.clear_name_cache() # Clear cache before creating new network
        bm.random.seed(SEED + 1)
        net_sig = BalancedNetwork(tau_d_I=tau)
        # Create injector that adds Frozen Signal on top of Background
        # We wrap the network to access it inside injector
        net_sig.network = net_sig 
        input_sig = FrozenPoissonInjector(net_sig, frozen_signal)
        
        runner_sig = bp.DSRunner(
            bp.DynSysGroup(net=net_sig, inp=input_sig),
            monitors={'spikes': net_sig.E.spike},
            dt=DT, progress_bar=False
        )
        runner_sig.run(duration)
        r_sig = np.mean(runner_sig.mon['spikes']) * 1000.0 / DT
        
        sens = compute_sensitivity(r_base, r_sig)
        print(f"  Sensitivity: {sens:.4f} (Base={r_base:.2f}Hz, Sig={r_sig:.2f}Hz)")

        # --- Part B: Reliability (Repeated Trials) ---
        print(f"  Measuring Reliability ({n_trials} trials)...")
        pop_spike_counts = [] # Store (Time,) for each trial
        
        for i in range(n_trials):
            # Reset network state but KEEP frozen signal same
            # Change random seed for Background Noise
            bp.math.clear_name_cache() # Clear cache before creating new network
            bm.random.seed(SEED + 100 + i) 
            
            net_rel = BalancedNetwork(tau_d_I=tau)
            net_rel.network = net_rel
            input_rel = FrozenPoissonInjector(net_rel, frozen_signal)
            
            runner_rel = bp.DSRunner(
                bp.DynSysGroup(net=net_rel, inp=input_rel),
                monitors={'spikes': net_rel.E.spike},
                dt=DT, progress_bar=False
            )
            runner_rel.run(duration)
            
            # Count population spikes per step
            # mon['spikes'] shape: (Steps, N_E) -> Sum to (Steps,)
            trial_pop_spikes = np.sum(runner_rel.mon['spikes'], axis=1)
            pop_spike_counts.append(trial_pop_spikes)
            
        # Shape: (n_trials, n_steps)
        all_trials_spikes = np.stack(pop_spike_counts)
        
        reliability = compute_reliability(
            all_trials_spikes, 
            bin_size_ms=50.0, 
            step_size_ms=20.0, 
            dt=DT
        )
        print(f"  Reliability: {reliability:.4f}")
        
        # Store
        results['tau'].append(tau)
        results['sensitivity'].append(sens)
        results['reliability'].append(reliability)

    return results

# ==========================================
# 2. Branching Model Trial (Control)
# ==========================================
def run_branching_model_experiment():
    print(f"\n{'='*60}")
    print(f"Running Branching Model (Control)")
    print(f"{'='*60}")
    
    # Branching parameters to scan (Sub -> Critical=1.0 -> Super)
    m_list = [0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05]
    duration = 1000
    n_trials = 100
    
    results = {'m': [], 'sensitivity': [], 'reliability': []}
    
    # Frozen Input for Branching (Sparse activation)
    # Ref Appendix B: Randomly activate neurons at Poisson times
    np.random.seed(SEED)
    frozen_input = (np.random.rand(duration, 1000) < 0.001).astype(float) # Sparse
    
    for m in m_list:
        print(f"Processing branching param m = {m}...")
        
        # 1. Sensitivity (Spontaneous vs Evoked)
        # Spontaneous (Input=0)
        bp.math.clear_name_cache() # Clear cache just in case
        bm.random.seed(SEED)
        bp_base = BranchingProcess(size=1000, branching_parameter=m, input_strength=0.0001) # Small noise
        runner_base = bp.DSRunner(bp_base, monitors={'state': bp_base.state}, dt=1.0)
        runner_base.run(duration)
        rate_base = np.mean(runner_base.mon['state'])
        
        # Evoked (Input = Frozen Signal)
        # Note: In branching model, input is usually added to probability
        # We need a custom runner loop or modified class to inject frozen input.
        # For simplicity, we assume the class handles `bp_base.input` update if we inject it.
        # Let's run a loop manually for the signal trial to inject input.
        
        state_sum_sig = 0
        bp.math.clear_name_cache() # Clear cache
        bp_sig = BranchingProcess(size=1000, branching_parameter=m, input_strength=0.0001)
        
        for t in range(duration):
            bp_sig.input.value += bm.asarray(frozen_input[t])
            bp_sig.update()
            state_sum_sig += bm.mean(bp_sig.state)
            
        rate_sig = state_sum_sig / duration
        sens = compute_sensitivity(rate_base, rate_sig)
        
        # 2. Reliability
        # Run multiple trials with SAME frozen_input but different internal stochasticity
        trial_activities = []
        
        for i in range(n_trials):
            bp.math.clear_name_cache() # Clear cache
            bm.random.seed(SEED + i)
            bp_rel = BranchingProcess(size=1000, branching_parameter=m, input_strength=0.0001)
            
            activity_trace = []
            for t in range(duration):
                bp_rel.input.value += bm.asarray(frozen_input[t])
                bp_rel.update()
                # Record total active nodes
                activity_trace.append(bm.sum(bp_rel.state))
            
            trial_activities.append(activity_trace)
            
        # Shape: (Trials, Time)
        all_trials = np.stack(trial_activities)
        rel = compute_reliability(all_trials, bin_size_ms=50, step_size_ms=20, dt=1.0)
        
        print(f"  Sens={sens:.3f}, Rel={rel:.3f}")
        results['m'].append(m)
        results['sensitivity'].append(sens)
        results['reliability'].append(rel)
        
    return results

if __name__ == "__main__":
    # 1. Run Balanced Network
    # Use a subset for quick testing, or full list for paper
    # target_taus = [2.0, 8.0, 11.0] 
    target_taus = TAU_DECAY_I_LIST 
    
    bn_data = run_balanced_network_experiment(target_taus)
    save_pkl(PROCESSED_DATA_DIR + "/sensitivity_reliability_bn.pkl", bn_data)
    
    # 2. Run Branching Model
    bp_data = run_branching_model_experiment()
    save_pkl(PROCESSED_DATA_DIR + "/sensitivity_reliability_bp.pkl", bp_data)
    
    print("\nExperiment Complete. Data saved to processed/ directory.")