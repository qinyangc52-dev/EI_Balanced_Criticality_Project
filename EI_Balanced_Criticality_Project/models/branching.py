import brainpy as bp
import brainpy.math as bm

class BranchingProcess(bp.DynamicalSystem):
    """
    Standard Branching Process Model for Control Experiment.
    Ref: Beggs & Plenz (2003), Yang et al. (2025) Appendix A.
    """
    def __init__(self, size, branching_parameter=1.0, input_strength=0.01):
        super().__init__()
        self.size = size
        self.m = branching_parameter  # Control parameter (sigma/lambda)
        self.h = input_strength       # External drive probability
        
        # State: 0 or 1 (Active)
        self.state = bm.Variable(bm.zeros(size, dtype=float))
        self.input = bm.Variable(bm.zeros(size, dtype=float))
        
        # Connection matrix (random)
        # To strictly strictly branching ratio m, weights are normalized
        self.conn = bm.random.rand(size, size) < (10.0 / size) # Sparse connectivity
        # Normalize so average outgoing sum is m
        self.W = self.conn * (self.m / 10.0) 

    def update(self):
        # 1. Internal Propagation
        # Probability of activation from neighbors
        # P(active) ~ sum(W * pre_state)
        prop_input = bm.matmul(self.state, self.W)
        
        # 2. Add External Input (Frozen or Random)
        total_prob = prop_input + self.input
        
        # 3. Stochastic Activation
        # Clip probability to [0, 1]
        prob = bm.clip(total_prob, 0.0, 1.0)
        
        # Determine next state
        self.state.value = bm.asarray(bm.random.rand(self.size) < prob, dtype=float)
        
        # Reset external input buffer
        self.input[:] = 0.0