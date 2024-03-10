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
from .utils import (
    stream_printer,
    check_solution_status,
    parse_dense_linear_constraints,
    minkowski_product,
)

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


class HyperbolicSVMHard(SVM):
    def __init__(
        self,
        lr: float = 1.0,
        seed: int = 1,
        batch_size: int = 128,
        epochs: int = 100,
        warm_start: bool = True,
        *kargs,
        **kwargs
    ):
        """
        :param warm_start: True to use Euclidean solution as a starting point, False use random initializations
        """
        super().__init__(*kargs, **kwargs)

        # training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.warm_start = warm_start

    def fit_binary(self, X: np.ndarray, y: np.ndarray, verbose=False, *kargs, **kwargs):
        """use gradient descent to solve the problem"""
        # initialize
        w = self._initialize(X, y, verbose=verbose)

        # TODO: implement projected SGD
        # TODO: we may translate from https://github.com/hhcho/hyplinear/blob/master/code/hsvm.m

    def _initialize(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """initialize guess of the decision function"""
        if self.warm_start:
            model = EuclideanSVMHard()  # use hard-margin for init
            if verbose:
                print("warm start init, solver log below:")
            model.fit(X, y, verbose=verbose)
            initial_w = model.w_
        else:
            raise NotImplementedError(
                "it is preferred to provide a warm start for this nonconvex problem"
            )
        return initial_w

    def _is_feasible(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> bool:
        """check if the current iterate is strictly feasible"""
        pass
        # TODO:

    def _projection(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """project to within the feasible region"""
        pass
        # TODO: implement projection


# * the following is not used, mosek refuses to solve nonconvex QP
# class HyperbolicSVMHard(SVM):
#     """original hyperbolic SVM formulation"""

#     def fit_binary(self, X: np.ndarray, y: np.ndarray, verbose=False, *kargs, **kwargs):
#         n, d = X.shape

#         # prepare linear constraints
#         G = np.eye(d)
#         G[0, 0] = -1.0
#         A = (X @ G) * y.reshape(-1, 1)
#         A_entries, A_rows, A_cols = parse_dense_linear_constraints(A)

#         # formulate problem
#         with mosek.Task() as task:
#             if verbose:
#                 task.set_Stream(mosek.streamtype.log, stream_printer)

#             # setup constraints and variables
#             bkc = [mosek.boundkey.lo] * (n + 1)
#             blc = [1.0] * n + [0.0]
#             buc = [+_INF] * (n + 1)
#             bkx = [mosek.boundkey.fr] * d
#             blx = [-_INF] * d
#             bux = [+_INF] * d
#             task.appendcons(n + 1)
#             task.appendvars(d)
#             task.putvarboundlist(range(d), bkx, blx, bux)

#             # specify entries in (-b^T w + 1 <= 0)
#             task.putaijlist(A_rows, A_cols, A_entries)
#             # specify quadratic constraint
#             task.putqconk(
#                 n,  # the last index of constraint as the quadratic one
#                 range(d),
#                 range(d),
#                 [-1.0] + [1.0] * (d - 1),
#             )
#             task.putconboundlist(range(n + 1), bkc, blc, buc)

#             # specify obj
#             task.putqobj(range(d), range(d), [-1.0] + [1.0] * (d - 1))  # diagonal 1s
#             task.putobjsense(mosek.objsense.minimize)

#             # run optimizer
#             task.optimize()

#             # check solution status
#             solsta = task.getsolsta(mosek.soltype.itr)
#             check_solution_status(solsta)

#             # get optimal solution
#             xx = task.getxx(mosek.soltype.itr)

#             # record solution
#             self.w_ = np.array(xx)

#             if verbose:
#                 task.solutionsummary(mosek.streamtype.msg)
#                 print("Optimal Solution: ")
#                 print("w: \n", self.w_)

#     def predict_binary(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
#         decision = (minkowski_product(X, self.w_) >= 0).astype(int)
#         return decision

#     def predict_multi(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
#         raise NotImplementedError()
