"""helper functions"""

# load packages
from typing import Tuple, List
import sys
import warnings

import numpy as np
from scipy.sparse import coo_array
import mosek


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
