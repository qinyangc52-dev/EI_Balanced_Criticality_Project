# core/frozen_input.py
"""
Frozen Input Injector with Background Noise
Combines deterministic signal with stochastic background for reliability testing
"""

import brainpy as bp
import brainpy.math as bm
from configs.model_config import (
    N_E, N_I, N_EXT, EXT_RATE,
    G_EXT_E, G_EXT_I, TAU_DECAY_E, V_REV_E, TAU_RISE
)


class FrozenSignalInjector(bp.DynamicalSystem):
    """
    Injects frozen signal + background noise.
    
    Key innovation: Separates deterministic signal (frozen) from 
    stochastic background (changes each trial), allowing us to measure 
    reliability (response consistency to frozen part) while maintaining 
    realistic network dynamics.
    """
    
    def __init__(self, network, frozen_signal_E, frozen_signal_I=None, 
                 bg_rate=None, signal_strength=1.0):
        """
        Args:
            network: Target BalancedNetwork instance
            frozen_signal_E: Pre-generated signal for E neurons (steps, N_E)
            frozen_signal_I: Pre-generated signal for I neurons (optional)
            bg_rate: Background Poisson rate (Hz), uses EXT_RATE if None
            signal_strength: Multiplier for signal strength (default=1.0)
        """
        super().__init__()
        
        self.network = network
        self.frozen_E = bm.asarray(frozen_signal_E, dtype=float)
        
        if frozen_signal_I is not None:
            self.frozen_I = bm.asarray(frozen_signal_I, dtype=float)
        else:
            # If no I signal provided, use zeros (signal only to E)
            self.frozen_I = bm.zeros((self.frozen_E.shape[0], N_I))
        
        self.bg_rate = bg_rate if bg_rate is not None else EXT_RATE
        self.signal_strength = signal_strength
        
        # Synaptic parameters (same as PoissonInput)
        self.tau_d = TAU_DECAY_E
        self.tau_r = TAU_RISE
        self.norm = 1.0 / (self.tau_d * self.tau_r)
        
        # State variables for background
        self.h_E_bg = bm.Variable(bm.zeros(N_E))
        self.g_E_bg = bm.Variable(bm.zeros(N_E))
        self.h_I_bg = bm.Variable(bm.zeros(N_I))
        self.g_I_bg = bm.Variable(bm.zeros(N_I))
        
        # State variables for signal
        self.h_E_sig = bm.Variable(bm.zeros(N_E))
        self.g_E_sig = bm.Variable(bm.zeros(N_E))
        self.h_I_sig = bm.Variable(bm.zeros(N_I))
        self.g_I_sig = bm.Variable(bm.zeros(N_I))
    
    def update(self):
        """Update both background and frozen signal components."""
        t_idx = bp.share['i']
        dt = bp.share['dt']
        
        # === Part 1: Background (Stochastic) ===
        p_spike = self.bg_rate * dt / 1000.0
        p_total = p_spike * N_EXT
        
        bg_E = bm.asarray(bm.random.rand(N_E) < p_total, dtype=float)
        bg_I = bm.asarray(bm.random.rand(N_I) < p_total, dtype=float)
        
        # Update background conductance
        self.h_E_bg.value = self.h_E_bg + (-self.h_E_bg / self.tau_r * dt + bg_E)
        self.g_E_bg.value = self.g_E_bg + (-self.g_E_bg / self.tau_d * dt + self.h_E_bg)
        
        self.h_I_bg.value = self.h_I_bg + (-self.h_I_bg / self.tau_r * dt + bg_I)
        self.g_I_bg.value = self.g_I_bg + (-self.g_I_bg / self.tau_d * dt + self.h_I_bg)
        
        # === Part 2: Frozen Signal (Deterministic) ===
        # Safe indexing
        idx = bm.minimum(t_idx, self.frozen_E.shape[0] - 1)
        
        sig_E = self.frozen_E[idx] * self.signal_strength
        sig_I = self.frozen_I[idx] * self.signal_strength
        
        # Update signal conductance
        self.h_E_sig.value = self.h_E_sig + (-self.h_E_sig / self.tau_r * dt + sig_E)
        self.g_E_sig.value = self.g_E_sig + (-self.g_E_sig / self.tau_d * dt + self.h_E_sig)
        
        self.h_I_sig.value = self.h_I_sig + (-self.h_I_sig / self.tau_r * dt + sig_I)
        self.g_I_sig.value = self.g_I_sig + (-self.g_I_sig / self.tau_d * dt + self.h_I_sig)
        
        # === Part 3: Combine and Inject ===
        # Total conductance
        g_E_total = (self.g_E_bg + self.g_E_sig) * G_EXT_E * self.norm
        g_I_total = (self.g_I_bg + self.g_I_sig) * G_EXT_I * self.norm
        
        # Compute currents
        I_ext_E = g_E_total * (V_REV_E - self.network.E.V)
        I_ext_I = g_I_total * (V_REV_E - self.network.I.V)
        
        # Add to network input
        self.network.E.input.value += I_ext_E
        self.network.I.input.value += I_ext_I


class BaselineInput(bp.DynamicalSystem):
    """
    Pure background noise (no signal) for baseline measurements.
    Essentially identical to PoissonInput.
    """
    
    def __init__(self, network, bg_rate=None):
        super().__init__()
        
        self.network = network
        self.rate = bg_rate if bg_rate is not None else EXT_RATE
        
        self.tau_d = TAU_DECAY_E
        self.tau_r = TAU_RISE
        self.norm = 1.0 / (self.tau_d * self.tau_r)
        
        self.h_E = bm.Variable(bm.zeros(N_E))
        self.g_E = bm.Variable(bm.zeros(N_E))
        self.h_I = bm.Variable(bm.zeros(N_I))
        self.g_I = bm.Variable(bm.zeros(N_I))
    
    def update(self):
        dt = bp.share['dt']
        
        p_spike = self.rate * dt / 1000.0
        p_total = p_spike * N_EXT
        
        bg_E = bm.asarray(bm.random.rand(N_E) < p_total, dtype=float)
        bg_I = bm.asarray(bm.random.rand(N_I) < p_total, dtype=float)
        
        self.h_E.value = self.h_E + (-self.h_E / self.tau_r * dt + bg_E)
        self.g_E.value = self.g_E + (-self.g_E / self.tau_d * dt + self.h_E)
        
        self.h_I.value = self.h_I + (-self.h_I / self.tau_r * dt + bg_I)
        self.g_I.value = self.g_I + (-self.g_I / self.tau_d * dt + self.h_I)
        
        g_E_norm = self.g_E * G_EXT_E * self.norm
        g_I_norm = self.g_I * G_EXT_I * self.norm
        
        I_ext_E = g_E_norm * (V_REV_E - self.network.E.V)
        I_ext_I = g_I_norm * (V_REV_E - self.network.I.V)
        
        self.network.E.input.value += I_ext_E
        self.network.I.input.value += I_ext_I