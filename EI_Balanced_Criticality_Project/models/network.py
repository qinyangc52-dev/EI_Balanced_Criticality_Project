# %% Stage 2: Model Components - E-I Balanced Network Assembly

import brainpy as bp
from configs.model_config import (
    N_E, N_I, CONN_PROB,
    G_EE, G_EI, G_IE, G_II,
    TAU_DECAY_E, V_REV_E, V_REV_I
)
from models.neurons import LifRefE, LifRefI
from models.synapses import DualExpCondSyn


class BalancedNetwork(bp.DynSysGroup):
    """E-I balanced network with 4 synaptic projections."""
    
    def __init__(self, tau_d_I: float, name='BalancedNet'):
        super().__init__(name=name)
        
        # Neuron populations
        self.E = LifRefE(N_E, name='E')
        self.I = LifRefI(N_I, name='I')
        
        # Excitatory synapses (AMPA dynamics)
        self.E2E = DualExpCondSyn(
            pre=self.E, post=self.E, prob=CONN_PROB,
            g_max=G_EE, tau_decay=TAU_DECAY_E, E_rev=V_REV_E,
            name='E2E'
        )
        
        self.E2I = DualExpCondSyn(
            pre=self.E, post=self.I, prob=CONN_PROB,
            g_max=G_EI, tau_decay=TAU_DECAY_E, E_rev=V_REV_E,
            name='E2I'
        )
        
        # Inhibitory synapses (GABA dynamics)
        self.I2E = DualExpCondSyn(
            pre=self.I, post=self.E, prob=CONN_PROB,
            g_max=G_IE, tau_decay=tau_d_I, E_rev=V_REV_I,
            name='I2E'
        )
        
        self.I2I = DualExpCondSyn(
            pre=self.I, post=self.I, prob=CONN_PROB,
            g_max=G_II, tau_decay=tau_d_I, E_rev=V_REV_I,
            name='I2I'
        )