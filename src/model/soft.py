"""
Soft SVM
- Euclidean SVM
- Hyperbolic SVM (original problem)
- Hyperbolic SVM (SDP relaxation)
- Hyperbolic SVM (SOS relaxation)
"""

# load packages
import numpy as np
import mosek

from .SVMBase import SVM
from .utils import (
    minkowski_product,
    stream_printer,
    check_solution_status,
    parse_sparse_linear_constraints,
)

# for symbolic placeholders
_INF = 0.0


class EuclideanSVMSoft(SVM):
    """treat hyperbolic data as living in the ambient Euclidean space"""

    def __init__(self, C: float = 1.0, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.C = C

    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
        k: int = 0,
        *kargs,
        **kwargs
    ):
        num_constraints, d = X.shape
        num_vars = d + 1 + num_constraints  # introduce \xi

        # get constraints, pack y(wx + b) + \xi >= 1
        A = np.hstack([X * y.reshape(-1, 1), y.reshape(-1, 1), np.eye(num_constraints)])
        A_entries, A_rows, A_cols = parse_sparse_linear_constraints(A)

        # formulate problem
        with mosek.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, stream_printer)

            # setup constraints and variables
            bkc = [mosek.boundkey.lo] * num_constraints
            blc = [1.0] * num_constraints
            buc = [+_INF] * num_constraints
            bkx = [mosek.boundkey.fr] * (d + 1) + [mosek.boundkey.lo] * num_constraints
            blx = [-_INF] * (d + 1) + [0.0] * num_constraints
            bux = [+_INF] * num_vars
            task.appendcons(num_constraints)
            task.appendvars(num_vars)
            task.putvarboundlist(range(num_vars), bkx, blx, bux)

            # specify entries in Ax >= b
            task.putaijlist(A_rows, A_cols, A_entries)
            task.putconboundlist(range(num_constraints), bkc, blc, buc)

            # specify obj
            task.putqobj(range(d), range(d), [1.0] * d)  # diagonal 1s
            task.putclist(
                range(d + 2, num_vars), [self.C] * num_constraints
            )  # plus \xi
            task.putobjsense(mosek.objsense.minimize)

            # run optimizer
            task.optimize()

            # check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            check_solution_status(solsta)

            # record solution
            xx = task.getxxslice(mosek.soltype.itr, 0, d + 1)
            w = np.array(xx[:-1])
            b = xx[-1]

            # append to self dict
            self._params[k] = (w, b)

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print("Optimal Solution: ")
                print("w: \n", self._params[k][0])
                print("b: \n", self._params[k][1])

    def decision_function(self, X: np.ndarray, k: int = 0):
        w, b = self._params[k]
        decision_vals = X @ w + b
        return decision_vals


class HyperbolicSVMSoftSDP(SVM):
    """Hyperbolic Soft-SVM, taken relaxation up to the first order"""

    def __init__(self, C: float = 1.0, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.C = C

    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
        k: int = 0,
        solution_type: str = "rank1",
        *kargs,
        **kwargs
    ):
        """
        after solving the relaxed model, to get back w, we could
        - "rank1": take rank1 decomposition of W
        - "rank1-gd": take rank1 decomposition and run a couple of projected SGD
        - "gaussian": draw random sample from N(z, W - zz^T) with the best in-sample classification performance
        """
        n, d = X.shape
        Z_dim = d + 1  # implement Z = [W, z; z^T, 1]

        # prepare linear constraints
        G = np.eye(d)
        G[0, 0] = -1.0

        # only specify lower triangular part, divide by 2
        A_flatten = (((X @ -G) * y.reshape(-1, 1)).flatten() / 2).tolist()

        # formulate problem
        with mosek.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, stream_printer)

            # formulate constraints (margin constraint + w feasible + bottom right = 1)
            bkc = [mosek.boundkey.lo] * (n + 1) + [mosek.boundkey.fx]
            blc = [1.0] * n + [0.0, 1.0]
            buc = [+_INF] * (n + 1) + [1.0]
            bkx = [mosek.boundkey.lo] * n
            blx = [0.0] * n
            bux = [+_INF] * n
            task.appendcons(n + 2)
            task.putconboundlist(range(n + 2), bkc, blc, buc)
            task.appendvars(n)  # \xi_i, the slack variables
            task.appendbarvars([Z_dim])
            task.putvarboundlist(range(n), bkx, blx, bux)  # \xi_i should be positive

            # add constraints to the system
            idxc = (
                np.hstack([np.ones((d,), dtype=int) * k for k in range(n)]).tolist()
                + [n] * d
                + [n + 1]
            )
            sdp_var_idx = [0] * len(idxc)  # all on the same SDP variable
            idx_k_list = [d] * (d * n) + list(range(d)) + [d]
            idx_l_list = list(range(d)) * n + list(range(d)) + [d]
            val_ijkl = A_flatten + [-1.0] + [1.0] * (d - 1) + [1.0]
            task.putbarablocktriplet(
                idxc, sdp_var_idx, idx_k_list, idx_l_list, val_ijkl
            )
            task.putaijlist(range(n), range(n), [2 ** (1 / 2)] * n)  # slack terms

            # add objective
            task.putbarcblocktriplet(
                [0] * d, range(d), range(d), [-1 / 2] + [1 / 2] * (d - 1)
            )
            task.putclist(range(n), [self.C] * n)  # add slack term penalization
            task.putobjsense(mosek.objsense.minimize)

            # run optimizer
            task.optimize()

            # check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            check_solution_status(solsta)

            # get optimal solution (only the lower triangular flattened is returned)
            bar_x = task.getbarxj(mosek.soltype.itr, 0)
            Z_bar_lower = np.array(bar_x)
            Z_bar = np.zeros((Z_dim, Z_dim))
            Z_bar[np.triu_indices(Z_dim)] = Z_bar_lower
            Z_bar += Z_bar.T
            Z_bar -= np.diag(np.diag(Z_bar)) / 2

            # record solution
            W_ = Z_bar[:, :-1][:-1, :]
            z_ = Z_bar[-1, :-1]

            self._params[k] = [W_, z_]

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print("Optimal Solution: ")
                print("W: \n", self._params[k][0])
                print("z: \n", self._params[k][1])

        # get w
        if solution_type == "rank1":
            # TODO: use LOBPCG to make this faster?
            # eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(W_)

            # take top rank 1 direction
            w_ = eigvecs[:, -1] * eigvals[-1] ** (1 / 2)
            self._params[k].append(w_)
        else:
            # TODO: implement other heuristics
            raise NotImplementedError()

    def decision_function(self, X: np.ndarray, k: int = 0):
        w = self._params[k][-1]
        decision_vals = minkowski_product(X, w)
        return decision_vals
