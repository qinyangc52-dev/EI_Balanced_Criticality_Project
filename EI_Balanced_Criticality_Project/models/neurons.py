# %% Stage 2: Model Components - LIF Neurons with Refractory Period
# RESTORED: Standard LIF implementation (Input divided by tau)

import brainpy as bp
import brainpy.math as bm
from configs.model_config import V_REST, V_RESET, V_TH, TAU_E, TAU_I, REF_E, REF_I

class LifRefE(bp.DynamicalSystem):
    def __init__(self, size, name=None):
        super().__init__(name=name)
        self.V_rest = V_REST
        self.V_reset = V_RESET
        self.V_th = V_TH
        self.tau = TAU_E
        self.t_ref = REF_E
        self.num = size
        self.V = bm.Variable(bm.ones(size) * V_REST)
        self.input = bm.Variable(bm.zeros(size))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)
    
    def update(self):
        t = bp.share['t']
        dt = bp.share['dt']
        refractory = (t - self.t_last_spike) <= self.t_ref
        # Standard LIF: Input current is divided by tau
        dV = ((self.V_rest - self.V) + self.input) / self.tau * dt
        V_new = bm.where(refractory, self.V, self.V + dV)
        spike = V_new >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V_new)
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.input[:] = 0.0

class LifRefI(bp.DynamicalSystem):
    def __init__(self, size, name=None):
        super().__init__(name=name)
        self.V_rest = V_REST
        self.V_reset = V_RESET
        self.V_th = V_TH
        self.tau = TAU_I
        self.t_ref = REF_I
        self.num = size
        self.V = bm.Variable(bm.ones(size) * V_REST)
        self.input = bm.Variable(bm.zeros(size))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)
    
    def update(self):
        t = bp.share['t']
        dt = bp.share['dt']
        refractory = (t - self.t_last_spike) <= self.t_ref
        dV = ((self.V_rest - self.V) + self.input) / self.tau * dt
        V_new = bm.where(refractory, self.V, self.V + dV)
        spike = V_new >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V_new)
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.input[:] = 0.0