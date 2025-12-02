# %% Stage 2: FIXED - E-I Balanced Network Assembly
# CRITICAL FIX: Explicit update order to prevent recursion

import brainpy as bp
from configs.model_config import (
    N_E, N_I, CONN_PROB,
    G_EE, G_EI, G_IE, G_II,
    TAU_DECAY_E, V_REV_E, V_REV_I
)
from models.neurons import LifRefE, LifRefI
from models.synapses import DualExpCondSyn


class BalancedNetwork(bp.DynSysGroup):
    """
    FIXED: E-I balanced network with explicit update control
    Problem: DynSysGroup auto-registers all attributes as nodes, causing recursion
    Solution: Control update order manually
    """
    
    def __init__(self, tau_d_I: float, name='BalancedNet'):
        super().__init__(name=name)
        
        # 1. Create neuron populations
        self.E = LifRefE(N_E, name='E')
        self.I = LifRefI(N_I, name='I')
        
        # 2. Create synapses (CRITICAL: use _syn prefix to prevent auto-registration)
        # By using underscore, we signal these are managed manually
        self._E2E = DualExpCondSyn(
            pre=self.E, post=self.E, prob=CONN_PROB,
            g_max=G_EE, tau_decay=TAU_DECAY_E, E_rev=V_REV_E,
            name='E2E'
        )
        
        self._E2I = DualExpCondSyn(
            pre=self.E, post=self.I, prob=CONN_PROB,
            g_max=G_EI, tau_decay=TAU_DECAY_E, E_rev=V_REV_E,
            name='E2I'
        )
        
        self._I2E = DualExpCondSyn(
            pre=self.I, post=self.E, prob=CONN_PROB,
            g_max=G_IE, tau_decay=tau_d_I, E_rev=V_REV_I,
            name='I2E'
        )
        
        self._I2I = DualExpCondSyn(
            pre=self.I, post=self.I, prob=CONN_PROB,
            g_max=G_II, tau_decay=tau_d_I, E_rev=V_REV_I,
            name='I2I'
        )
        
        # 3. Store synapses in a list for manual iteration
        self.synapses = [self._E2E, self._E2I, self._I2E, self._I2I]
    
    def update(self):
        """
        CRITICAL: Manual update order to prevent recursion
        Order: Neurons -> Synapses (in single pass)
        """
        # Step 1: Update neurons first (this clears input buffers)
        self.E.update()
        self.I.update()
        
        # Step 2: Update all synapses (they add to input buffers for NEXT step)
        for syn in self.synapses:
            syn.update()


# Alternative: Use bp.Network instead of DynSysGroup
class BalancedNetwork_Alt(bp.Network):
    """
    Alternative implementation using bp.Network
    This might handle update order better
    """
    
    def __init__(self, tau_d_I: float, name='BalancedNet'):
        # Create components first
        self.E = LifRefE(N_E, name='E')
        self.I = LifRefI(N_I, name='I')
        
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
        
        # Pass to Network with explicit order
        super().__init__(
            self.E, self.I,           # Neurons first
            self.E2E, self.E2I,       # Then synapses
            self.I2E, self.I2I,
            name=name
        )


# SAFEST: Manual update with no auto-registration
class BalancedNetwork_Manual(bp.DynamicalSystem):
    """
    Safest implementation: Complete manual control
    No automatic node discovery
    """
    
    def __init__(self, tau_d_I: float, name='BalancedNet'):
        super().__init__(name=name)
        
        # Create neurons
        self.E = LifRefE(N_E, name='E')
        self.I = LifRefI(N_I, name='I')
        
        # Create synapses (NOT auto-registered as children)
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
    
    def update(self):
        """Manual two-phase update"""
        # Phase 1: Update neurons (uses input from PREVIOUS step, then clears)
        self.E.update()
        self.I.update()
        
        # Phase 2: Update synapses (computes input for NEXT step)
        self.E2E.update()
        self.E2I.update()
        self.I2E.update()
        self.I2I.update()
    
    def reset(self):
        """Reset all components"""
        self.E.__init__(N_E, name='E')
        self.I.__init__(N_I, name='I')
        # Synapses will auto-reset their state variables