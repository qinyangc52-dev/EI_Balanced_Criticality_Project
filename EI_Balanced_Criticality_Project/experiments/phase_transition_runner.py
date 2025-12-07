# %% Stage 3: Experiments - Main Simulation Loop for Phase Transition
# FIXED: Memory Overflow Fix using Chunked Simulation
# FIXED: Random seed reset for reproducibility across tau values
# Iterates over tau_d_I values and records spike trains

import brainpy as bp
import brainpy.math as bm
import numpy as np
import gc  # Garbage Collector
from pathlib import Path
from configs.model_config import (
    TAU_DECAY_I_LIST, DT, DURATION, WARMUP,
    RAW_DATA_DIR, SEED
)
from models.network import BalancedNetwork
from core.inputs import PoissonInput
from utils.io_manager import save_npz


class PhaseTransitionRunner:
    """
    Runs simulations across different tau_d_I values to observe phase transition.
    Includes memory optimization for long simulations.
    """
    
    def __init__(self, tau_d_I_list=None, seed=None):
        self.tau_d_I_list = tau_d_I_list if tau_d_I_list else TAU_DECAY_I_LIST
        self.seed = seed if seed is not None else SEED
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    def run_single_tau(self, tau_d_I: float):
        """
        Run simulation for a single tau_d_I value.
        Uses CHUNKED execution to prevent Out-Of-Memory errors during long runs.
        
        FIXED: Added proper state cleanup and random seed reset to ensure
        reproducibility and prevent state leakage between different tau runs.
        """
        print(f"\n{'='*60}")
        print(f"Running simulation: tau_d_I = {tau_d_I:.1f} ms")
        print(f"{'='*60}")
        
        # ========== CRITICAL FIX: Complete State Reset ==========
        # 1. Clear BrainPy internal caches
        bm.clear_name_cache()
        
        # 2. Reset random seed (CRITICAL for reproducibility)
        # This ensures each tau value starts with the same random sequence
        bm.random.seed(self.seed)
        np.random.seed(self.seed)
        
        # 3. Reset global time variables
        bp.share.save(t=0.0, dt=DT, i=0)
        
        # 4. Force garbage collection
        gc.collect()
        # ========================================================
        
        # 1. Setup Network and Input
        tau_name = str(tau_d_I).replace('.', '_')
        net = BalancedNetwork(tau_d_I=tau_d_I, name=f'BalancedNet_{tau_name}')
        ext_input = PoissonInput(net, name=f'PoissonInput_{tau_name}')
        
        class SimSystem(bp.DynSysGroup):
            def __init__(self):
                super().__init__()
                self.net = net
                self.ext = ext_input
            
            def update(self):
                self.ext.update()
                self.net.update()
        
        system = SimSystem()

        # 2. Configure Chunking
        # Run 5 seconds at a time (safe for memory)
        chunk_duration = 5000.0 
        total_duration = DURATION
        n_chunks = int(np.ceil(total_duration / chunk_duration))
        
        print(f"Simulating {total_duration} ms in {n_chunks} chunks (dt={DT} ms)...")
        
        all_spike_times = []
        all_neuron_indices = []
        
        # Initialize time
        current_time = 0.0
        bp.share.save(t=0.0, dt=DT) # Ensure global time starts at 0

        # 3. Main Chunk Loop
        for i in range(n_chunks):
            # Determine length of this chunk (handle last chunk remainder)
            time_remaining = total_duration - current_time
            run_dur = min(chunk_duration, time_remaining)
            
            if run_dur <= 0: break

            # Create a FRESH runner for each chunk to reset monitor memory
            # Note: The 'system' (net) preserves its state (V, g, etc.) across chunks
            runner = bp.DSRunner(
                system,
                monitors={'net.E.spike': net.E.spike},
                dt=DT,
                progress_bar=False  # Disable individual progress bars to reduce noise
            )
            
            # CRITICAL: Sync runner time with simulation time
            # BrainPy runners usually start at i=0 (t=0). We must update 'i' manually
            # to ensure 't' continues correctly for refractory periods etc.
            runner.i = int(round(current_time / DT))
            bp.share.save(t=current_time)
            
            # Run Chunk
            runner.run(run_dur)
            
            # 4. Extract & Process Data immediately (freeing JAX memory)
            # Convert to numpy immediately
            chunk_spikes = runner.mon['net.E.spike']
            spike_indices = bm.where(chunk_spikes)
            
            # Move to CPU/Numpy
            t_indices = np.array(spike_indices[0])
            n_indices = np.array(spike_indices[1])
            
            if len(t_indices) > 0:
                # Calculate absolute time: (step_in_chunk * dt) + chunk_start_time
                real_times = t_indices * DT + current_time
                
                # Filter out Warmup period
                if current_time < WARMUP:
                    valid_mask = real_times >= WARMUP
                    real_times = real_times[valid_mask]
                    n_indices = n_indices[valid_mask]
                
                # Store if not empty
                if len(real_times) > 0:
                    all_spike_times.append(real_times)
                    all_neuron_indices.append(n_indices)
            
            # Update time
            current_time += run_dur
            
           # Progress Log
            progress = (i + 1) / n_chunks * 100
            print(f"  Chunk {i+1}/{n_chunks}: {run_dur:.0f}ms done. Time: {current_time:.0f}ms ({progress:.1f}%)")
            
            # 5. Clean up Memory
            del runner
            gc.collect()  # Force Python GC

        # 6. Concatenate Results
        if len(all_spike_times) > 0:
            final_spike_times = np.concatenate(all_spike_times)
            final_neuron_indices = np.concatenate(all_neuron_indices)
        else:
            final_spike_times = np.array([])
            final_neuron_indices = np.array([])
        
        # Stats
        print(f"Total E spikes (post-warmup): {len(final_spike_times)}")
        effective_duration = DURATION - WARMUP
        if effective_duration > 0 and len(final_spike_times) > 0:
            rate = len(final_spike_times) / effective_duration / 800 * 1000
            print(f"Mean firing rate: {rate:.2f} Hz")
        else:
            print("Warning: No spikes detected!")
        
        return final_spike_times, final_neuron_indices
    
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