# %% Stage 2: SAFEST - Dual Exponential Synapse
# Complete rewrite to avoid ALL potential recursion sources

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp
import numpy as np
from configs.model_config import TAU_RISE


class DualExpCondSyn(bp.DynamicalSystem):
    """
    SAFEST: Minimal inheritance, pure functional style
    Changed from bp.Projection to bp.DynamicalSystem
    """
    
    def __init__(self, pre, post, prob, g_max, tau_decay, E_rev, name=None):
        super().__init__(name=name)
        
        # Store neuron references (NOT as children)
        self.pre = pre
        self.post = post
        
        # Parameters
        self.g_max = g_max
        self.tau_r = TAU_RISE
        self.tau_d = tau_decay
        self.E_rev = E_rev
        self.norm = 1.0 / (self.tau_d * self.tau_r)
        
        # Generate connectivity with pure numpy (avoid BrainPy random)
        mask = np.random.rand(pre.num, post.num) < prob
        self.conn_mask = jnp.array(mask, dtype=jnp.float32)
        
        # State variables
        self.h = bm.Variable(bm.zeros(post.num))
        self.g = bm.Variable(bm.zeros(post.num))
        
        # Cache dt at initialization
        self._dt = None
    
    def update(self):
        """
        Pure functional update - no recursion possible
        """
        # Get dt once
        if self._dt is None:
            self._dt = bp.share['dt']
        dt = self._dt
        
        try:
            # Extract all values as pure JAX arrays immediately
            # This prevents any lazy evaluation or property access recursion
            pre_spike = jnp.array(self.pre.spike.value, dtype=jnp.float32)
            h_curr = jnp.array(self.h.value, dtype=jnp.float32)
            g_curr = jnp.array(self.g.value, dtype=jnp.float32)
            post_v = jnp.array(self.post.V.value, dtype=jnp.float32)
            post_input = jnp.array(self.post.input.value, dtype=jnp.float32)
            
            # Pure JAX computation
            spike_input = jnp.dot(pre_spike, self.conn_mask)
            
            # Explicit Euler integration
            h_new = h_curr + dt * (-h_curr / self.tau_r) + spike_input
            g_new = g_curr + dt * (-g_curr / self.tau_d) + h_curr
            
            # Synaptic current
            g_syn = g_new * self.g_max * self.norm
            I_syn = g_syn * (self.E_rev - post_v)
            
            # Accumulate input
            post_input_new = post_input + I_syn
            
            # Single atomic updates (no incremental operations)
            self.h.value = h_new
            self.g.value = g_new
            self.post.input.value = post_input_new
            
        except Exception as e:
            print(f"Synapse {self.name} update failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def reset_state(self):
        """Reset synapse state"""
        self.h[:] = 0.0
        self.g[:] = 0.0


# Debug version with recursion detection
class DualExpCondSyn_Debug(bp.DynamicalSystem):
    """
    Use this to diagnose recursion issues
    """
    _call_stack = []
    MAX_DEPTH = 5
    
    def __init__(self, pre, post, prob, g_max, tau_decay, E_rev, name=None):
        super().__init__(name=name)
        
        self.pre = pre
        self.post = post
        self.g_max = g_max
        self.tau_r = TAU_RISE
        self.tau_d = tau_decay
        self.E_rev = E_rev
        self.norm = 1.0 / (self.tau_d * self.tau_r)
        
        mask = np.random.rand(pre.num, post.num) < prob
        self.conn_mask = jnp.array(mask, dtype=jnp.float32)
        
        self.h = bm.Variable(bm.zeros(post.num))
        self.g = bm.Variable(bm.zeros(post.num))
        self._dt = None
    
    def update(self):
        # Track call stack
        call_id = f"{self.name}:{id(self)}"
        
        if call_id in DualExpCondSyn_Debug._call_stack:
            print(f"⚠️ RECURSION DETECTED!")
            print(f"   Call stack: {' -> '.join(DualExpCondSyn_Debug._call_stack)}")
            print(f"   Trying to call: {call_id}")
            return
        
        if len(DualExpCondSyn_Debug._call_stack) > self.MAX_DEPTH:
            print(f"⚠️ MAX DEPTH EXCEEDED!")
            print(f"   Call stack: {' -> '.join(DualExpCondSyn_Debug._call_stack)}")
            return
        
        DualExpCondSyn_Debug._call_stack.append(call_id)
        
        try:
            if self._dt is None:
                self._dt = bp.share['dt']
            dt = self._dt
            
            pre_spike = jnp.array(self.pre.spike.value, dtype=jnp.float32)
            h_curr = jnp.array(self.h.value, dtype=jnp.float32)
            g_curr = jnp.array(self.g.value, dtype=jnp.float32)
            post_v = jnp.array(self.post.V.value, dtype=jnp.float32)
            post_input = jnp.array(self.post.input.value, dtype=jnp.float32)
            
            spike_input = jnp.dot(pre_spike, self.conn_mask)
            h_new = h_curr + dt * (-h_curr / self.tau_r) + spike_input
            g_new = g_curr + dt * (-g_curr / self.tau_d) + h_curr
            
            g_syn = g_new * self.g_max * self.norm
            I_syn = g_syn * (self.E_rev - post_v)
            post_input_new = post_input + I_syn
            
            self.h.value = h_new
            self.g.value = g_new
            self.post.input.value = post_input_new
            
        finally:
            DualExpCondSyn_Debug._call_stack.pop()