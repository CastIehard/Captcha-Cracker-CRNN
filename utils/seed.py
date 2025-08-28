import torch
import numpy as np

def set_seed(seed: int):
    """
    Set random seed for reproducibility across torch and numpy.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # If GPU is available, also set seed for CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
