# %% Quick Test: Verify firing rate after parameter fix
# Run 200ms simulation to check if network has 1-5 Hz firing rate (not 487 Hz)

import brainpy as bp
import brainpy.math as bm
import numpy as np
from configs.model_config import (
    N_E, N_I, DT, SEED, TAU_DECAY_I_CRITICAL
)
from models.network import BalancedNetwork
from core.inputs import PoissonInput


def quick_test_firing_rate():
    """Test firing rate with corrected parameters."""
    
    print("\n" + "="*70)
    print("QUICK TEST: Firing Rate Validation (200ms)")
    print("="*70 + "\n")
    
    # Clear cache
    bm.clear_name_cache()
    bm.random.seed(SEED)
    
    # Build network with critical tau value
    net = BalancedNetwork(tau_d_I=TAU_DECAY_I_CRITICAL, name='quick_test')
    
    # Add external Poisson input
    ext_input = PoissonInput(net)
    
    # Create combined system  
    class QuickTestSystem(bp.DynSysGroup):
        def __init__(self):
            super().__init__()
            self.net = net
            self.ext = ext_input
        
        def update(self):
            self.ext.update()
            self.net.update()
    
    system = QuickTestSystem()
    
    # Run simulation (200ms)
    duration = 200.0  # ms
    runner = bp.DSRunner(
        system,
        monitors={'net.E.spike': net.E.spike, 'net.I.spike': net.I.spike},
        dt=DT,
        progress_bar=True
    )
    
    print(f"Running {duration} ms simulation...")
    runner.run(duration)
    
    # Calculate firing rates
    spikes_e = runner.mon['net.E.spike']  # shape: (T_steps, N_E)
    spikes_i = runner.mon['net.I.spike']  # shape: (T_steps, N_I)
    
    n_spikes_e = spikes_e.sum()
    n_spikes_i = spikes_i.sum()
    n_spikes_total = n_spikes_e + n_spikes_i
    
    # Firing rates in Hz
    fr_e = (n_spikes_e / N_E) / (duration / 1000.0)  # Hz
    fr_i = (n_spikes_i / N_I) / (duration / 1000.0)  # Hz
    fr_pop = n_spikes_total / (N_E + N_I) / (duration / 1000.0)  # Hz
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"Total spikes:")
    print(f"  Excitatory (E): {int(n_spikes_e)} spikes in {N_E} neurons")
    print(f"  Inhibitory (I): {int(n_spikes_i)} spikes in {N_I} neurons")
    print(f"  Total: {int(n_spikes_total)} spikes")
    
    print(f"\nFiring rates:")
    print(f"  E-neurons: {fr_e:.2f} Hz")
    print(f"  I-neurons: {fr_i:.2f} Hz")
    print(f"  Population: {fr_pop:.2f} Hz")
    
    print(f"\nTarget range: 1-5 Hz")
    if 1.0 <= fr_pop <= 5.0:
        print(f"✓ PASS: Firing rate is within target range!")
    else:
        print(f"✗ FAIL: Firing rate is outside target range")
        if fr_pop > 5.0:
            print(f"  Current: {fr_pop:.2f} Hz (too high)")
        else:
            print(f"  Current: {fr_pop:.2f} Hz (too low)")
    
    print("\n" + "="*70 + "\n")
    
    return fr_pop


if __name__ == "__main__":
    quick_test_firing_rate()
