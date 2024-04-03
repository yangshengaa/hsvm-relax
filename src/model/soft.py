"""
Soft SVM
- Euclidean SVM
- Hyperbolic SVM (original problem)
- Hyperbolic SVM (SDP relaxation)
- Hyperbolic SVM (SOS relaxation)
"""

# load packages
import warnings
import numpy as np
import mosek
import scipy
import sympy as sp
from SumOfSquares import poly_opt_prob

from .SVMBase import SVM
from .utils import (
    stream_printer,
    check_solution_status,
    parse_sparse_linear_constraints,
    minkowski_product,
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
        **kwargs,
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
        *kargs,
        **kwargs,
    ):
        n, d = X.shape
        Z_dim = d + 1  # implement Z = [W, z; z^T, 1]

        # prepare linear constraints
        G = np.eye(d)
        G[0, 0] = -1.0

        # only specify lower triangular part, divide by 2
        B = (X @ -G) * y.reshape(-1, 1)
        A_flatten = (B.flatten() / 2).tolist()

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

            # get w using heuristic methods
            w_ = self._get_optima(W_, z_, X, y)
            solution_value = (-w_[0] ** 2 + w_[1] ** 2) / 2 + np.clip(
                np.arcsinh(1) - np.arcsinh(B @ w_), a_min=0.0, a_max=None
            ).sum() * self.C
            self._params[k] = [W_, z_, w_]

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print("Optimal Solution: ")
                print("W: \n", self._params[k][0])
                print("z: \n", self._params[k][1])
                print("w: \n", self._params[k][2])
                print(f"Optimal Value: {solution_value:.4f}")

    def decision_function(self, X: np.ndarray, k: int = 0):
        w = self._params[k][-1]
        decision_vals = minkowski_product(X, w)
        return decision_vals

    def _get_optima(
        self,
        W: np.ndarray,
        z: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        num_random: int = 10,
    ) -> np.ndarray:
        """
        heuristic methods to extract the optimal w from (W, z) solved
        we search for the following candidates
        - z
        - rank 1 component of W
        - columns divided by z (from MIT Robust Optimization)
        - Gaussian Randomization (from Stanford Lecture)

        in soft margin we take the solution that has the lowest overall cost only
        here we take the arcsinh original formulation

        :param W, z: obtained by solver
        :param X, y: given data
        :param num_random
        :return w: the optima classification boundary
        """
        # z
        candidates = [z]

        # rank 1 component
        eigvals, eigvecs = np.linalg.eigh(W)
        w_r1 = eigvecs[:, -1] * eigvals[-1] ** (1 / 2)
        candidates.append(w_r1)

        # column divisions
        candidates.append((W / z).T)

        # gaussian randomization N(z, W - zz^T)
        random_solutions = (
            np.random.normal(0, 1, size=(num_random, len(z)))
            @ np.linalg.cholesky(W - np.outer(z, z)).T
            + z
        )
        candidates.append(random_solutions)

        candidates = np.vstack(candidates)

        # vectorized slackness computations
        d = X.shape[1]
        G = np.eye(d)
        G[0, 0] = -1.0
        B = (X @ -G) * y.reshape(-1, 1)
        # filter feasible candidates
        feasible = (candidates * (candidates @ G)).sum(axis=-1) >= 0
        candidates = candidates[feasible]
        slackness = np.clip(
            np.arcsinh(1) - np.arcsinh(B @ candidates.T), a_min=0.0, a_max=None
        )
        costs = (candidates * (candidates @ G)).sum() / 2 + self.C * slackness.sum(
            axis=0
        )

        # get candidate with the min cost
        w = candidates[np.argmin(costs)]
        return w


class HyperbolicSVMSoft(SVM):
    def __init__(
        self,
        C: float = 1.0,
        lr: float = 1.0,
        seed: int = 1,
        batch_size: int = 128,
        epochs: int = 100,
        warm_start: bool = True,
        *kargs,
        **kwargs,
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

        # penalization strength
        self.C = C

    def fit_binary(
        self, X: np.ndarray, y: np.ndarray, verbose=False, k: int = 0, *kargs, **kwargs
    ):
        """use gradient descent to solve the problem"""
        # initialize
        w = self._initialize(X, y, verbose=verbose)
        if not self._is_feasible(w):
            w = self._projection(w, alpha=0.01)

        # train
        # TODO: add minibatch
        w_new = w
        best_w = w
        init_loss = self._loss_fn(w, X, y, self.C)
        min_loss = init_loss
        for _ in range(self.epochs):
            current_loss = 0
            # full batch GD
            grad_w = self._grad_fn(w_new, X, y)
            w_new = w_new - self.lr * grad_w
            # if not in feasible region, need to use projection
            if not self._is_feasible(w_new):
                # solve optimization problem for nearest feasible point
                alpha_opt = self._alpha_search(w_new)
                # project w to feasible sub-space
                w_new = self._projection(w_new, alpha_opt)
            current_loss = self._loss_fn(w_new, X, y, self.C)

            # update loss and estimate
            if current_loss < min_loss:
                min_loss = current_loss
                best_w = w_new

        self._params[k] = best_w
        solution_value = self._loss_fn(best_w, X, y, self.C)

        if verbose:
            print("Optimal Solution: ")
            print("w: \n", self._params[k])
            print(f"Solution Value: {solution_value:.4f}")

    def _initialize(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """initialize guess of the decision function"""
        if self.warm_start:
            model = EuclideanSVMSoft()  # use hard-margin for init
            if verbose:
                print("warm start init, solver log below:")
            model.fit(X, y, verbose=verbose)
            initial_w = model._params[0][0]
        else:
            raise NotImplementedError(
                "it is preferred to provide a warm start for this nonconvex problem"
            )
        return initial_w

    def _is_feasible(self, w: np.ndarray) -> bool:
        """check if the current iterate is strictly feasible"""
        feasibility = -w[0] ** 2 + np.dot(w[1:], w[1:]).item()
        return feasibility > 0

    def _projection(self, w: np.ndarray, alpha: float, eps: float = 1e-6) -> np.ndarray:
        """project w to within the boundary"""
        proj_w = w.copy()
        proj_w[1:] = (1 + alpha) * proj_w[1:]
        first_sgn = 1 if proj_w[0] >= 0 else -1
        proj_w[[0]] = first_sgn * np.sqrt(np.sum(proj_w[1:] ** 2) - eps)
        return proj_w

    def _alpha_search(self, w: np.ndarray) -> float:
        """
        use scipy to solve for alpha in projection
        """
        res = scipy.optimize.minimize_scalar(
            lambda alpha: np.sum((self._projection(w, alpha) - w) ** 2)
        )
        alpha = res.x
        return alpha

    def _loss_fn(
        self, w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 1
    ) -> float:
        """compute the loss function, with l1 penalty and squared hinge loss"""
        w = w.reshape(-1, 1)
        loss_term = 1 / 2 * (-np.square(w[0, 0]) + np.dot(w[1:].T, w[1:]).item())
        misclass_term = y.reshape(-1, 1) * (-w[[0]] * X[:, [0]] + X[:, 1:] @ w[1:])
        misclass_loss = np.arcsinh(1.0) - np.arcsinh(-misclass_term)
        loss = loss_term + C * np.sum(np.where(misclass_loss > 0, misclass_loss, 0))
        return loss

    def _grad_fn(
        self, w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 1
    ) -> np.ndarray:
        """compute gradient of the loss function"""
        w = w.reshape(-1, 1)
        grad_margin = np.vstack((-w[[0]], w[1:]))
        z = y.reshape(-1, 1) * (-w[[0]] * X[:, [0]] + X[:, 1:] @ w[1:])
        misclass = (np.arcsinh(1.0) - np.arcsinh(-z)) > 0
        arcsinh_term = -1 / np.sqrt(z**2 + 1)
        mink_prod_term = y.reshape(-1, 1) * np.hstack((X[:, [0]], -X[:, 1:]))
        grad_misclass = misclass * arcsinh_term * mink_prod_term
        grad_w = grad_margin + C * np.sum(grad_misclass, axis=0, keepdims=True).T
        grad_w = grad_w.flatten()
        return grad_w

    def decision_function(self, X: np.ndarray, k: int = 0):
        w = self._params[k]
        decision_vals = minkowski_product(X, w)
        return decision_vals

class HyperbolicSVMSoftSOS(SVM):
    def __init__(self, C: float = 1.0, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.C = C

    def fit_binary(self, X: np.ndarray, y: np.ndarray, verbose=False, k: int = 0, max_kappa=10, *kargs, **kwargs):
        # get problem size
        n, dim = X.shape

        # formulate problem
        w = sp.symbols(f"w0:{dim}")
        xi = sp.symbols(f"xi1:{n+1}")
        decision_vars = (*w, *xi)

        obj = ((-w[0] ** 2 + sum(w[i] ** 2 for i in range(1, dim))) / 2 +
            self.C * sum(xi)
        )
        # add inequality constraints
        inequalities = []
        for i in range(n):
            cur_inequality = y[i] * (X[i][0] * w[0] - sum(X[i][k] * w[k] for k in range(1, dim))) + np.sqrt(2) * xi[i] - 1
            inequalities.append(cur_inequality)
            inequalities.append(xi[i])
        inequalities.append(-w[0] ** 2 + sum(w[k] ** 2 for k in range(1, dim)))
        # no equality constraints
        equalities = []
        
        # formulate problem 
        for kappa in range(3, max_kappa):
            prob = poly_opt_prob(decision_vars, obj, eqs=equalities, ineqs=inequalities, deg=kappa)
            try:
                # solve the problem
                prob.solve(solver="mosek")

                # solution found, exit loop
                print(prob.value)
                break
            except:
                warnings.warn(
                    f"relaxation order kappa = {kappa} failed, increase by 1")
