"""
Hard SVM
- Euclidean SVM
- Hyperbolic SVM (original problem)
- Hyperbolic SVM (SDP relaxation)
- Hyperbolic SVM (SOS relaxation)
"""

# load packages
import numpy as np
import mosek

from .SVMBase import SVM
from .utils import stream_printer, check_solution_status, parse_dense_linear_constraints

# for symbolic placeholders
_INF = 0.0


class EuclideanSVMHard(SVM):
    """treat hyperbolic data as living in the ambient Euclidean space"""

    def fit_binary(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False, *kargs, **kwargs
    ):
        num_constraints, d = X.shape

        # get constraints, pack y(wx + b) >= 1 into A(w, b) >= 1 and flatten
        A = np.hstack([X * y.reshape(-1, 1), y.reshape(-1, 1)])
        A_entries, A_rows, A_cols = parse_dense_linear_constraints(A)

        # formulate problem
        with mosek.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, stream_printer)

            # setup constraints and variables
            bkc = [mosek.boundkey.lo] * num_constraints
            blc = [1.0] * num_constraints
            buc = [+_INF] * num_constraints
            bkx = [mosek.boundkey.fr] * (d + 1)
            blx = [-_INF] * (d + 1)
            bux = [+_INF] * (d + 1)
            task.appendcons(num_constraints)
            task.appendvars(d + 1)
            task.putvarboundlist(range(d + 1), bkx, blx, bux)

            # specify entries in Ax >= b
            task.putaijlist(A_rows, A_cols, A_entries)
            task.putconboundlist(range(num_constraints), bkc, blc, buc)

            # specify obj
            task.putqobj(range(d), range(d), [1.0] * d)  # diagonal 1s
            task.putobjsense(mosek.objsense.minimize)

            # run optimizer
            task.optimize()

            # check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            check_solution_status(solsta)

            # get optimal solution
            xx = task.getxx(mosek.soltype.itr)

            # record solution
            self.w_ = np.array(xx[:-1])
            self.b_ = xx[-1]

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print("Optimal Solution: ")
                print("w: \n", self.w_)
                print("b: \n", self.b_)

    def predict_binary(self, X: np.ndarray, *kargs, **kwargs):
        decision = ((X @ self.w_ + self.b_) >= 0).astype(int)
        return decision

    def predict_multi(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        raise NotImplementedError()
