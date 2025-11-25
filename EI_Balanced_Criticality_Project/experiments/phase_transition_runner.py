# %% Stage 3: Experiments - Main Simulation Loop for Phase Transition
# Iterates over tau_d_I values and records spike trains

import brainpy as bp
import brainpy.math as bm
import numpy as np
from pathlib import Path
from configs.model_config import (
    TAU_DECAY_I_LIST, DT, DURATION, WARMUP,
    RAW_DATA_DIR
)
from models.network import BalancedNetwork
from core.inputs import PoissonInput
from utils.io_manager import save_npz


class PhaseTransitionRunner:
    """
    Runs simulations across different tau_d_I values to observe phase transition.
    """
    
    def __init__(self, tau_d_I_list=None):
        self.tau_d_I_list = tau_d_I_list if tau_d_I_list else TAU_DECAY_I_LIST
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    def run_single_tau(self, tau_d_I: float):
        """
        Run simulation for a single tau_d_I value.
        Returns spike times and neuron indices for E population.
        """
        print(f"\n{'='*60}")
        print(f"Running simulation: tau_d_I = {tau_d_I:.1f} ms")
        print(f"{'='*60}")
        
        # Clear name cache to avoid naming conflicts in multi-trial runs
        bm.clear_name_cache()
        
        # [修改点] 将 2.0 转换为 2_0，防止 BrainPy 报错
        tau_name = str(tau_d_I).replace('.', '_')
        
        # Build network with unique SAFE name
        net = BalancedNetwork(tau_d_I=tau_d_I, name=f'BalancedNet_{tau_name}')
        
        # Add external input (建议也给个名字，方便调试)
        ext_input = PoissonInput(net, name=f'PoissonInput_{tau_name}')
        
        # Create combined system
        class SimSystem(bp.DynSysGroup):
            def __init__(self):
                super().__init__()
                self.net = net
                self.ext = ext_input
            
            def update(self):
                # Order matters: external input first, then network dynamics
                self.ext.update()
                self.net.update()
        
        system = SimSystem()
        
        # Create runner with spike monitoring
        runner = bp.DSRunner(
            system,
            monitors={'net.E.spike': net.E.spike},
            dt=DT,
            progress_bar=True
        )
        
        # Run simulation
        print(f"Simulating {DURATION} ms with dt={DT} ms...")
        runner.run(DURATION)
        
        # Extract spike data
        spike_mat = runner.mon['net.E.spike']  # shape: (T_steps, N_E)
        
        # Convert to spike times format (discard warmup)
        warmup_steps = int(WARMUP / DT)
        spike_mat_valid = spike_mat[warmup_steps:]
        
        # Find spike times and neuron indices
        spike_indices = bm.where(spike_mat_valid)
        
        if len(spike_indices[0]) > 0:
            t_indices = np.array(spike_indices[0])
            neuron_indices = np.array(spike_indices[1])
            spike_times = t_indices * DT + WARMUP
        else:
            spike_times = np.array([])
            neuron_indices = np.array([])
        
        print(f"Total E spikes: {len(spike_times)}")
        if len(spike_times) > 0:
            print(f"Mean firing rate: {len(spike_times) / (DURATION - WARMUP) / 800 * 1000:.2f} Hz")
        else:
            print("Warning: No spikes detected!")
        
        return spike_times, neuron_indices
    
    def run_all(self):
        """
        Run simulations for all tau_d_I values in the list.
        Save raw spike data for each.
        """
        results = {}
        
        for tau_d_I in self.tau_d_I_list:
            spike_times, neuron_indices = self.run_single_tau(tau_d_I)
            
            # Save raw data
            filename = f"spikes_{tau_d_I:.1f}.npz"
            filepath = Path(RAW_DATA_DIR) / filename
            
            save_npz(
                filepath,
                spike_times=spike_times,
                neuron_indices=neuron_indices,
                tau_d_I=tau_d_I,
                duration=DURATION - WARMUP
            )
            
            results[tau_d_I] = {
                'spike_times': spike_times,
                'neuron_indices': neuron_indices,
                'file': str(filepath)
            }
            
            print(f"Saved raw data to {filepath}")
        
        print(f"\n{'='*60}")
        print(f"All simulations completed!")
        print(f"{'='*60}\n")
        
        return results
