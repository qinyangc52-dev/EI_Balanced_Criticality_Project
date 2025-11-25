# %% Stage 2: Model Components - Dual Exponential Conductance-Based Synapse
# Implements equation (A2, A3) from paper

import brainpy as bp
import brainpy.math as bm
from configs.model_config import TAU_RISE


class DualExpCondSyn(bp.Projection):
    """
    Dual-exponential conductance-based synapse (COBA).
    """
    
    def __init__(
        self,
        pre: bp.DynamicalSystem,
        post: bp.DynamicalSystem,
        prob: float,
        g_max: float,
        tau_decay: float,
        E_rev: float,
        name: str = None
    ):
        super().__init__(name=name)
        
        self.pre = pre
        self.post = post
        
        self.g_max = g_max
        self.tau_r = TAU_RISE
        self.tau_d = tau_decay
        self.E_rev = E_rev
        
        # Build connection using BrainPy's standard approach
        self.comm = bp.dnn.EventCSRLinear(
            bp.connect.FixedProb(prob, pre=pre.num, post=post.num),
            weight=1.0
        )
        
        # State variables
        self.h = bm.Variable(bm.zeros(post.num))
        self.g = bm.Variable(bm.zeros(post.num))
        
        # Normalization factor
        self.norm = 1.0 / (self.tau_d - self.tau_r)
    
    def update(self):
        # Get pre-synaptic spikes and propagate through connection
        pre_spike = bm.asarray(self.pre.spike.value, dtype=float)
        spike_input = self.comm(pre_spike)
        
        # Update h (rise dynamics)
        dh = -self.h / self.tau_r * bp.share['dt'] + spike_input
        self.h.value = self.h + dh
        
        # Update g (decay dynamics)
        dg = -self.g / self.tau_d * bp.share['dt'] + self.h
        self.g.value = self.g + dg
        
        # Compute synaptic current
        g_normalized = self.g * self.g_max * self.norm
        I_syn = g_normalized * (self.E_rev - self.post.V)
        
        # Add to post-synaptic current
        self.post.input.value = self.post.input + I_syn