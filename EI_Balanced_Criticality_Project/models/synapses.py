# %% Stage 2: Model Components - Dual Exponential Synapse
# RESTORED: Paper Eq A3 Normalization

import brainpy as bp
import brainpy.math as bm
from configs.model_config import TAU_RISE

class DualExpCondSyn(bp.Projection):
    def __init__(self, pre, post, prob, g_max, tau_decay, E_rev, name=None):
        super().__init__(name=name)
        self.pre = pre   #前突触神经元群（提供spike输入）
        self.post = post
        self.g_max = g_max
        self.tau_r = TAU_RISE
        self.tau_d = tau_decay
        self.E_rev = E_rev
        
        self.comm = bp.dnn.EventCSRLinear(
            bp.connect.FixedProb(prob, pre=pre.num, post=post.num),
            weight=1.0
        )
        self.h = bm.Variable(bm.zeros(post.num))
        self.g = bm.Variable(bm.zeros(post.num))
        
        # Standard Normalization from Paper Eq A3
        # Ensures the integral of the kernel is 1.0 (Unit Charge)
        self.norm = 1.0 / (self.tau_d * self.tau_r)
    
    def update(self):
        pre_spike = bm.asarray(self.pre.spike.value, dtype=float)
        spike_input = self.comm(pre_spike)
        
        dh = -self.h / self.tau_r * bp.share['dt'] + spike_input
        self.h.value = self.h + dh
        
        dg = -self.g / self.tau_d * bp.share['dt'] + self.h
        self.g.value = self.g + dg
        
        g_normalized = self.g * self.g_max * self.norm
        I_syn = g_normalized * (self.E_rev - self.post.V)
        self.post.input.value = self.post.input + I_syn