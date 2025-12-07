# experiments/frozen_signal_generator.py
"""
Frozen Signal Generator for Reliability Testing
Generates deterministic input patterns that can be replayed across trials
"""

import numpy as np
from pathlib import Path
from configs.model_config import N_E, N_I, DT, SEED, PROCESSED_DATA_DIR
from utils.io_manager import save_pkl, load_pkl


class FrozenSignalGenerator:
    """
    Generate and manage frozen Poisson signals for reliability experiments.
    
    Key concept: A "frozen signal" is a pre-generated spike pattern that 
    remains identical across multiple trials, allowing us to measure how 
    reliably the network responds to the same input.
    """
    
    def __init__(self, duration_ms=2000.0, signal_freq=25.0, seed=SEED):
        """
        Args:
            duration_ms: Duration of signal in milliseconds
            signal_freq: Frequency of signal spikes (Hz)
            seed: Random seed for reproducibility
        """
        self.duration = duration_ms
        self.freq = signal_freq
        self.seed = seed
        self.steps = int(duration_ms / DT)
        
    def generate_signal(self, target_population='E'):
        """
        Generate a single frozen signal pattern.
        
        Args:
            target_population: 'E' or 'I'
            
        Returns:
            signal: Binary array of shape (time_steps, N_neurons)
        """
        np.random.seed(self.seed)
        
        n_neurons = N_E if target_population == 'E' else N_I
        prob_spike = self.freq * DT / 1000.0
        
        # Generate binary spike pattern
        signal = (np.random.rand(self.steps, n_neurons) < prob_spike).astype(float)
        
        return signal
    
    def generate_signal_pair(self):
        """
        Generate two different frozen signals for classification task.
        
        Returns:
            signal_A, signal_B: Two distinct signal patterns
        """
        # Signal A
        np.random.seed(self.seed)
        signal_A = self.generate_signal('E')
        
        # Signal B (different seed)
        np.random.seed(self.seed + 1000)
        signal_B = self.generate_signal('E')
        
        return signal_A, signal_B
    
    def save_signals(self, filename='frozen_signals.pkl'):
        """Save generated signals to disk for reuse."""
        signals = {
            'signal_E': self.generate_signal('E'),
            'signal_I': self.generate_signal('I'),
            'duration': self.duration,
            'freq': self.freq,
            'steps': self.steps
        }
        
        filepath = Path(PROCESSED_DATA_DIR) / filename
        save_pkl(filepath, signals)
        print(f"Frozen signals saved to {filepath}")
        
        return signals
    
    @staticmethod
    def load_signals(filename='frozen_signals.pkl'):
        """Load pre-generated signals from disk."""
        filepath = Path(PROCESSED_DATA_DIR) / filename
        return load_pkl(filepath)


# Quick test
if __name__ == "__main__":
    gen = FrozenSignalGenerator(duration_ms=2000.0, signal_freq=25.0)
    signals = gen.save_signals()
    
    print(f"Generated frozen signal:")
    print(f"  Shape: {signals['signal_E'].shape}")
    print(f"  Total spikes: {np.sum(signals['signal_E'])}")
    print(f"  Avg rate: {np.sum(signals['signal_E']) / signals['duration'] / N_E * 1000:.2f} Hz")