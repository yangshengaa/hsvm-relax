"""real world dataset"""

# load packages
import os
from typing import Tuple
import numpy as np


def load_lorentz_features(name: str, path: str) -> Tuple[np.ndarray]:
    """load lorentz features from precomputed lorentz embeddings"""
    load_path = os.path.join(path, name)
    with open(os.path.join(load_path, "X_lorentz.npy"), "rb") as f:
        X = np.load(f)
    with open(os.path.join(load_path, "y.npy"), "rb") as f:
        y = np.load(f)

    return X, y
