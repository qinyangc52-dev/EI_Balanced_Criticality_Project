# %% Utilities - I/O Manager for Data Persistence
# Centralized save/load functions for npz and pkl formats

import numpy as np
import pickle
from pathlib import Path


def save_npz(filepath, **kwargs):
    """
    Save data to .npz format.
    
    Args:
        filepath: Path object or string
        **kwargs: key-value pairs to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, **kwargs)


def load_npz(filepath):
    """
    Load data from .npz format.
    
    Returns:
        NpzFile object (dict-like)
    """
    return np.load(filepath)


def save_pkl(filepath, data):
    """
    Save data to .pkl format using pickle.
    
    Args:
        filepath: Path object or string
        data: any picklable Python object
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(filepath):
    """
    Load data from .pkl format.
    
    Returns:
        Unpickled Python object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_dir(dirpath):
    """
    Ensure directory exists, create if not.
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)