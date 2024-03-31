"""helper functions"""

# load packages
from typing import Tuple, List
import sys
import warnings

import numpy as np
from scipy.sparse import coo_array
import mosek


# =============================================================
# --------------------- mosek related -------------------------
# =============================================================


# for mosek printing
def stream_printer(text):
    sys.stdout.write(text)
    sys.stdout.flush()


# check mosek optimization result
def check_solution_status(solsta: mosek.solsta):
    """check solution status. If not optimal, throw a warning"""
    if solsta == mosek.solsta.optimal:
        pass
    elif solsta == mosek.solsta.dual_infeas_cer:
        warnings.warn("Primal or dual infeasibility!!!\n")
    elif solsta == mosek.solsta.prim_infeas_cer:
        warnings.warn("Primal or dual infeasibility!!!\n")
    elif mosek.solsta.unknown:
        warnings.warn("Unknown solution status!!!")
    else:
        warnings.warn("Other solution status!!!")


def parse_dense_linear_constraints(
    A: np.ndarray,
) -> Tuple[List[float], List[int], List[int]]:
    """keep dense representation but convert to (val, row, col) format"""
    n, p = A.shape
    A_entries = A.flatten()
    A_rows = np.hstack([np.ones((p,), dtype=int) * n for n in range(n)])
    A_cols = np.arange(p).tolist() * n
    return A_entries, A_rows, A_cols


def parse_sparse_linear_constraints(
    A: np.ndarray,
) -> Tuple[List[float], List[int], List[int]]:
    """
    turn a dense representation of linear constraint matrix into (val, row, col) representations
    """
    A_sparse = coo_array(A)
    return A_sparse.data, A_sparse.row, A_sparse.col


# =================================================================
# --------------- hyperbolic util functions -----------------------
# =================================================================


def minkowski_product(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    inner product on Lorentz manifold, assume R^(1, n) layout
    :param X: either a 1D array or a 2D matrix
    :param w: assume 1D (decision plane)
    """
    p = X.shape[1]

    # metric tensor
    G = np.eye(p)
    G[0, 0] = -1

    # inner product
    result = -X @ G @ w
    return result

def get_decision_boundary(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    get the center and radius of the circle marking the decision boundary given by w (on Poincare space)
    """
    w0 = w[0]
    wr = w[1:]
    center = wr / w0
    radius = np.sqrt((wr ** 2).sum() / w0 ** 2 - 1)
    return center, radius


# ================================================================
# ------------ platt scaling related -----------------------------
# ================================================================
# Modified from http://www.work.caltech.edu/~htlin/program/libsvm/doc/platt.py


def get_platt_scaling_coef(
    decision_vals: np.ndarray,
    labels: np.ndarray,
    prior0: float = None,
    prior1: float = None,
    max_iteration: int = 1000,
    min_step: float = 1e-10,
    sigma: float = 1e-12,
    eps: float = 1e-5,
) -> Tuple[float]:
    """
    the training process of platt

    :param max_iteration: maximum iterations to train platt
    :param min_step: the minimum stepsize in the line search
    :param sigma: for numerical PD
    :param eps: convergence guarantee threshold
    :return a, b: two scalars parameters in the logistic regression
    """
    # Count prior0 and prior1 if needed
    n = len(labels)
    if prior0 is None and prior1 is None:
        prior1 = sum(labels > 0)
        prior0 = n - prior1

    # count target support
    hi_target = (prior1 + 1.0) / (prior1 + 2.0)
    lo_target = 1 / (prior0 + 2.0)
    t = np.where(labels > 0, hi_target, lo_target)

    # initial point and initial function value
    a, b = 0, np.log((prior0 + 1) / (prior1 + 1))

    fapb = decision_vals * a + b
    t_transformed = np.where(fapb >= 0, t, t - 1)
    fval = np.sum(t_transformed * fapb + np.log(1 + np.exp(-np.abs(fapb))))

    for _ in range(max_iteration):
        # update gradient and hessian
        h11 = h22 = sigma
        h21 = g1 = g2 = 0

        fapb = decision_vals * a + b

        p = np.where(fapb >= 0, np.exp(-fapb), 1) / (1 + np.exp(-np.abs(fapb)))
        q = 1 - p

        d2 = p * q
        h11 += np.sum(decision_vals * decision_vals * d2)
        h22 += np.sum(d2)
        h21 += np.sum(decision_vals * d2)

        d1 = t - p
        g1 += np.sum(decision_vals * d1)
        g2 += np.sum(d1)

        # stopping criterion
        if abs(g1) < eps and abs(g2) < eps:
            break

        # finding Newton's Direction
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        # line seearch
        stepsize = 1
        while stepsize >= min_step:
            new_a = a + stepsize * dA
            new_b = b + stepsize * dB

            # new function values
            fapb = decision_vals * new_a + new_b
            t_transformed = np.where(fapb >= 0, t, t - 1)
            new_f = np.sum(t_transformed * fapb + np.log(1 + np.exp(-np.abs(fapb))))

            # check sufficient decrease
            if new_f < fval + 0.0001 * stepsize * gd:
                a, b, fval = new_a, new_b, new_f
                break
            else:
                stepsize = stepsize / 2.0

        # failure management
        if stepsize < min_step:
            print("line search failed")
            return a, b

    return a, b


def platt_decision(decision_vals: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    make decision based on platt scaling

    :param decision_vals: the raw decision values
    :param a, b: the scalars trained
    :return: the probability of belonging to a certain class (binarized)
    """
    fapb = decision_vals * a + b
    decisions = np.where(fapb >= 0, np.exp(-fapb), 1) / (1 + np.exp(-np.abs(fapb)))
    return decisions
