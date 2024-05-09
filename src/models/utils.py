"""helper functions"""

# load packages
from typing import Tuple, List
import sys
import warnings
from itertools import combinations_with_replacement

import numpy as np
from scipy.sparse import coo_array
from scipy.optimize import minimize
import mosek
import sympy
from sympy.core.symbol import Symbol


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
    radius = np.sqrt((wr**2).sum() / w0**2 - 1)
    return center, radius


def logmap0_lorentz(X: np.ndarray) -> np.ndarray:
    """the log map at origin of Lorentz manifold"""
    x0 = X[:, [0]]
    xr = X[:, 1:]
    x_norm = np.arccosh(x0)
    v = xr * x_norm / np.sinh(x_norm)
    return v


def get_jacobian_batch(X: np.ndarray) -> np.ndarray:
    """
    for robust counterpart, get jacobian with respect to input X
    :return [num_samples, d, d - 1]
    """
    n, d = X.shape
    xr = X[:, 1:]
    V = logmap0_lorentz(X)

    # assemble jacobian for each input
    jacobian_batch = []
    for i in range(n):
        cur_x = xr[[i]]
        cur_v = V[[i]]
        v_norm = np.linalg.norm(cur_v)
        bottom = (
            np.cosh(v_norm) / v_norm - np.sinh(v_norm) / v_norm**3
        ) * cur_v.T @ cur_v + np.sinh(v_norm) / v_norm * np.eye(d - 1)
        cur_jacobian = np.vstack([cur_x, bottom])
        jacobian_batch.append(cur_jacobian)

    jacobian_batch = np.array(jacobian_batch)
    return jacobian_batch


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


# ===============================================================
# -------------------- SOSTools ---------------------------------
# ===============================================================
def monomials(vartable: List[Symbol], degree: List[int]) -> List[Symbol]:
    """
    adapted from https://github.com/zm2404/SOSPy

    Construct a vector of monomials with prespecified degrees.
    Z = monomials(VARTABLE, DEGREE)

    Given a list of independent variables VARTABLE and a list of non-negative
    integers DEGREE, this function constructs a column vector
    containing all possible monomials of degree described in DEGREE.

    For example, monomials([x1,x2],[1,2,3]) with x1 and x2
    being symbolic variables will return exponent matrix of monomials in x1 and
    x2 of degree 1, 2, and 3.
    """
    num_variables = len(vartable)

    Z = []

    for d in degree:
        for comb in combinations_with_replacement(range(num_variables), d):
            z = np.zeros(num_variables, dtype=int)
            for idx in comb:
                z[idx] += 1
            Z.append(z)

    res = []
    for row in Z:
        exprs = [var**deg for var, deg in zip(vartable, row)]
        res.append(sympy.prod(exprs))
    return res


def svec2smat_batch(matrix_svec: np.ndarray) -> np.ndarray:
    """
    convert a svec moment matrix multiplier into the original moment matrix multiplier
    :return a moment matrix multiplier with batch dimension as the last dimension
    """
    # check that length is of (n + 1) * n / 2 form
    delta = np.sqrt(1 + 8 * matrix_svec.shape[0])
    assert int(delta) == delta
    d = (-1 + int(delta)) // 2

    #
    mat_list = []
    for n in range(matrix_svec.shape[1]):
        # extract column
        arr = matrix_svec[:, n]

        # put to upper triangular parts
        cur_mat = np.zeros((d, d))
        cur_mat[np.triu_indices(d)] = arr
        cur_mat += cur_mat.T

        # account for sqrt(2)
        template = np.ones_like(cur_mat)
        template -= np.eye(template.shape[0])
        template /= np.sqrt(2)
        template += np.eye(template.shape[0]) / 2
        cur_mat = cur_mat * template

        mat_list.append(cur_mat)

    mat = np.stack(mat_list, axis=0)
    return mat


# ===================================
# -------- local refinement ---------
# ===================================
def objective_soft(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 1.0
) -> float:
    """
    the objective for soft-margin HSVM
    :param w: the decision boundary
    :param X: the data matrix
    :param y: the data labels (binarized)
    :param C: the slack penalty
    :return the loss
    """
    return (
        -1 / 2 * minkowski_product(w.reshape(1, -1), w).item()
        + C
        * np.sum(
            np.clip(np.arcsinh(1) - np.arcsinh(y * minkowski_product(X, w)), 0, np.inf)
        ).item()
    )


def local_refinement(
    w0: np.ndarray, X: np.ndarray, y: np.ndarray, C: float, method="COBYLA"
) -> np.ndarray:
    """
    perform local refinement of the solutions given by SDP/SOS using minimize
    :param w0: the solutions given by relaxed methods
    :param X: the data matrix
    :param y: the data labels (binarized)
    :param C: the slack penalization
    :param method: the refinement method
    :return the refined result
    """
    # method could be COBYLA, SLSQP, and trust-constr
    assert method in [
        "COBYLA",
        "SLSQP",
        "trust-constr",
    ], f'method {method} not available for constrained opt, select ["COBYLA", "SLSQP", "trust-constr"]'

    cons = {"type": "ineq", "fun": lambda w: -minkowski_product(w.reshape(1, -1), w)}
    res = minimize(
        fun=objective_soft, x0=w0, args=(X, y, C), method=method, constraints=cons
    )
    new_w = res.x
    return new_w
