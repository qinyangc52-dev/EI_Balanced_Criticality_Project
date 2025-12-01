# %% Stage 2: Core Components - External Poisson Input
# FIXED: Normalization factor and removed double counting

import brainpy as bp
import brainpy.math as bm
from configs.model_config import (
    N_E, N_I, N_EXT, EXT_RATE,
    G_EXT_E, G_EXT_I, TAU_DECAY_E, V_REV_E, TAU_RISE
)

class PoissonInput(bp.DynamicalSystem):
    """External Poisson spike input to both E and I populations."""
    
    def __init__(self, network, name='PoissonInput'):
        super().__init__(name=name)
        
        self.network = network
        self.rate = EXT_RATE
        self.N_ext = N_EXT
        
        self.g_ext_E = G_EXT_E
        self.g_ext_I = G_EXT_I
        
        self.tau_r = TAU_RISE
        self.tau_d = TAU_DECAY_E
        self.E_rev = V_REV_E
        
        # FIXED NORMALIZATION:
        self.norm = 1.0 / (self.tau_d * self.tau_r)
        
        # State variables
        self.h_E = bm.Variable(bm.zeros(N_E))
        self.g_E = bm.Variable(bm.zeros(N_E))
        self.h_I = bm.Variable(bm.zeros(N_I))
        self.g_I = bm.Variable(bm.zeros(N_I))
    
    def update(self):
        dt = bp.share['dt']
        
        # Poisson spike generation
        p_spike = self.rate * dt / 1000.0
        p_total = p_spike * self.N_ext
        
        # Generate boolean spikes
        ext_spikes_E = bm.random.rand(N_E) < p_total
        ext_spikes_I = bm.random.rand(N_I) < p_total
        
        ext_E = bm.asarray(ext_spikes_E, dtype=float) 
        ext_I = bm.asarray(ext_spikes_I, dtype=float)
        
        # Update E conductance
        self.h_E.value = self.h_E + (-self.h_E / self.tau_r * dt + ext_E)
        self.g_E.value = self.g_E + (-self.g_E / self.tau_d * dt + self.h_E)
        
        # Apply synaptic weight
        g_E_norm = self.g_E * self.g_ext_E * self.norm
        I_ext_E = g_E_norm * (self.E_rev - self.network.E.V)
        
        # Update I conductance
        self.h_I.value = self.h_I + (-self.h_I / self.tau_r * dt + ext_I)
        self.g_I.value = self.g_I + (-self.g_I / self.tau_d * dt + self.h_I)
        
        g_I_norm = self.g_I * self.g_ext_I * self.norm
        I_ext_I = g_I_norm * (self.E_rev - self.network.I.V)
        
        # Add to network
        self.network.E.input.value = self.network.E.input + I_ext_E
        self.network.I.input.value = self.network.I.input + I_ext_I