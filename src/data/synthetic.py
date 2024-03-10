"""
synthetic dataset
- Gaussian mixture + lifting
"""

# load packages
from typing import Tuple
import numpy as np

# prevent division by zero
_EPS = 1e-16


def hyperbolic_gaussian(
    dim=2, K: int = 2, n: int = 100, seed: int = 0, scale: float = 1.0
) -> Tuple[np.ndarray]:
    """
    generate guassian mixture on lorentz manifold
    :param dim: the dimension of the hyperbolic data, living in dim + 1 Euclidean space
    :param n: the number of samples per class
    :param K: number of classes
    :param seed: the random seed
    :param scale: the scale of gaussian data
    :return (X, y)
    """
    # fix seed
    np.random.seed(seed)

    # geneerate center
    centers = np.random.randn(K, dim)
    Xs, ys = [], []
    for k in range(K):
        mu = centers[[k]]

        # generate random covariance structure by cholesky
        # R = np.triu(np.random.randn(dim, dim))

        # isotropic
        R = np.eye(dim)

        cur_X = np.random.normal(size=(n, dim), scale=scale) @ R + mu
        Xs.append(cur_X)
        ys.append(np.ones((n,)) * k)

    X = np.vstack(Xs)
    X = expmap0_lorentz(X)  # lift
    y = np.vstack(ys)

    return X, y


def expmap0_lorentz(x: np.ndarray) -> np.ndarray:
    """
    lorentz manifold exponential map at the origin
    :param x: input data in the Euclidean space
    :param c: the negative curvature, assume ( > 0 )
    :return lifted hyperbolic features
    """
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x0 = np.cosh(x_norm)
    xr = np.sinh(x_norm) * x / (x_norm + _EPS)
    mapped_x = np.hstack([x0, xr])
    return mapped_x


def stereo_l2p(x: np.ndarray) -> np.ndarray:
    """
    stereographic projection from lorentz to poincare, for visualization purpose
    """
    features = x[:, 1:] / (1 + x[:, [0]])
    return features
