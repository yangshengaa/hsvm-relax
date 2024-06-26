"""
Hard SVM
- Euclidean SVM
- Hyperbolic SVM (original problem)
- Hyperbolic SVM (SDP relaxation)
- Hyperbolic SVM (SOS relaxation)
"""

# load packages
import warnings
import sympy as sp
import scipy
import numpy as np
import mosek
from SumOfSquares import poly_opt_prob

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
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
        k: int = 0,
        *kargs,
        **kwargs,
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
            w = np.array(xx[:-1])
            b = xx[-1]

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


class HyperbolicSVMHard(SVM):
    def __init__(
        self,
        lr: float = 1.0,
        seed: int = 1,
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
        self.epochs = epochs
        self.seed = seed
        self.warm_start = warm_start

    def fit_binary(
        self, X: np.ndarray, y: np.ndarray, verbose=False, k: int = 0, *kargs, **kwargs
    ):
        """use gradient descent to solve the problem"""
        # initialize
        w = self._initialize(X, y, verbose=verbose)

        if not self._is_feasible(w, X, y):
            w = self._projection(w, X, y)

        # train
        w_new = w
        best_w = w
        init_loss = self._loss_fn(w, X, y)
        min_loss = init_loss
        for _ in range(self.epochs):
            current_loss = 0
            # full batch GD
            grad_w = self._grad_fn(w_new, X, y)
            w_new = w_new - self.lr * grad_w
            # if not in feasible region, need to use projection
            if not self._is_feasible(w_new, X, y):
                # solve optimization problem for nearest feasible point
                w_new = self._projection(w_new, X, y)
            current_loss = self._loss_fn(w_new, X, y)

            # update loss and estimate
            if current_loss < min_loss:
                min_loss = current_loss
                best_w = w_new

        self._params[k] = best_w
        solution_value = self._loss_fn(best_w, X, y)

        if verbose:
            print("Optimal Solution: ")
            print("w: \n", self._params[k])
            print(f"Solution Value: {solution_value:.4f}")

    def _initialize(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """initialize guess of the decision function"""
        if self.warm_start:
            model = EuclideanSVMHard()  # use hard-margin for init
            if verbose:
                print("warm start init, solver log below:")
            model.fit(X, y, verbose=verbose)
            initial_w = model._params[0][0]
        else:
            raise NotImplementedError(
                "it is preferred to provide a warm start for this nonconvex problem"
            )
        return initial_w

    def _is_feasible(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> bool:
        """check if the current iterate is strictly feasible"""
        d = X.shape[1]
        hyperbolic_boundary = (-w[0] ** 2 + np.dot(w[1:], w[1:]).item()) < 0
        G = np.eye(d)
        G[0, 0] = -1
        B = (X @ -G) * y.reshape(-1, 1)
        separable = (B @ w >= 1).all()

        return hyperbolic_boundary and separable

    def _projection(self, wt: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """project to within the feasible region"""
        num_constraints, d = X.shape

        G = np.eye(d)
        G[0, 0] = -1
        B = (X @ -G) * y.reshape(-1, 1)
        B_entries, B_rows, B_cols = parse_dense_linear_constraints(B)
        # formulate problem
        with mosek.Task() as task:
            # setup constraints and variables
            bkc = [mosek.boundkey.lo] * num_constraints
            blc = [1.0] * num_constraints
            buc = [+_INF] * num_constraints
            bkx = [mosek.boundkey.fr] * d
            blx = [-_INF] * d
            bux = [+_INF] * d
            task.appendcons(num_constraints)
            task.appendvars(d)
            task.putvarboundlist(range(d), bkx, blx, bux)

            # add constraints
            task.putaijlist(B_rows, B_cols, B_entries)
            task.putconboundlist(range(num_constraints), bkc, blc, buc)

            # specify obj
            task.putqobj(range(d), range(d), [1] * d)
            task.putclist(range(d), -wt)
            task.putobjsense(mosek.objsense.minimize)

            # run optimizer
            task.optimize()

            # check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            check_solution_status(solsta)

            # get optimal solution
            xx = np.array(task.getxx(mosek.soltype.itr))

        return xx

    def _loss_fn(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """compute the loss function, with l1 penalty and squared hinge loss"""
        loss = 1 / 2 * (-np.square(w[0]) + np.dot(w[1:].T, w[1:]).item())
        return loss

    def _grad_fn(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """compute gradient of the loss function"""
        grad_w = np.hstack((-w[0], w[1:]))
        return grad_w

    def decision_function(self, X: np.ndarray, k: int = 0):
        w = self._params[k]
        decision_vals = minkowski_product(X, w)
        return decision_vals


class HyperbolicSVMHardSDP(SVM):
    def __init__(self, redundant=False, *kargs, **kwargs):
        """
        :param redundant: add redundant constraint to make better estimation of W
        trick from MIT Robust Optimization
        """
        super().__init__(*kargs, **kwargs)
        self.redundant = redundant

    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose=False,
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

        # process redundant constraints
        if self.redundant:
            val_ijkl_red = []
            tril_rows, tril_cols = np.tril_indices(Z_dim)
            idx_k_list_red = tril_rows.tolist() * n
            idx_l_list_red = tril_cols.tolist() * n
            idxc_red = np.hstack(
                [
                    np.ones((len(tril_rows),), dtype=int) * k
                    for k in range(n + 2, 2 * n + 2)
                ]
            ).tolist()
            # append values
            for i in range(n):
                # parse redundant constraints
                b_i = B[[i]].T
                redundant_mat = np.vstack(
                    (np.hstack((b_i @ b_i.T, b_i)), np.hstack((b_i.T, np.ones((1, 1)))))
                )
                val_ijkl_red += redundant_mat[(tril_rows, tril_cols)].tolist()

        # formulate problem
        with mosek.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, stream_printer)

            # formulate constraints (margin constraint + w feasible + bottom right = 1)
            bkc = [mosek.boundkey.lo] * (n + 1) + [mosek.boundkey.fx]
            blc = [1.0] * n + [0.0, 1.0]
            buc = [+_INF] * (n + 1) + [1.0]
            if self.redundant:
                bkc += [mosek.boundkey.lo] * n
                blc += [0.0] * n
                buc += [+_INF] * n
            num_constraints = len(bkc)
            task.appendcons(num_constraints)
            task.putconboundlist(range(num_constraints), bkc, blc, buc)
            task.appendbarvars([Z_dim])

            # add constraints to the system
            idxc = (
                np.hstack([np.ones((d,), dtype=int) * k for k in range(n)]).tolist()
                + [n] * d
                + [n + 1]
            )
            idx_k_list = [d] * (d * n) + list(range(d)) + [d]
            idx_l_list = list(range(d)) * n + list(range(d)) + [d]
            val_ijkl = A_flatten + [-1.0] + [1.0] * (d - 1) + [1.0]
            # add redundant
            if self.redundant:
                idxc += idxc_red
                idx_k_list += idx_k_list_red
                idx_l_list += idx_l_list_red
                val_ijkl += val_ijkl_red
            sdp_var_idx = [0] * len(idxc)  # all on the same SDP variable
            # communicate with task at once
            task.putbarablocktriplet(
                idxc, sdp_var_idx, idx_k_list, idx_l_list, val_ijkl
            )

            # add objective
            task.putbarcblocktriplet(
                [0] * d, range(d), range(d), [-1 / 2] + [1 / 2] * (d - 1)
            )
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

            # get w
            w_ = self._get_optima(W_, z_, X, y)
            solution_value = (w_ * (G @ w_)).sum() / 2
            self._params[k] = [W_, z_, w_]

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print("Optimal Solution: ")
                print("W: \n", self._params[k][0])
                print("z: \n", self._params[k][1])
                print(f"Solution Value: {solution_value:.4f}")

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

        in hard margin we take the solution that is feasible and has the lowest cost
        if none satisfy all constraints, we take the rank1 component

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

        # vectorized feasibility check
        d = X.shape[1]
        G = np.eye(d)
        G[0, 0] = -1.0
        B = (X @ -G) * y.reshape(-1, 1)
        separable = (B @ candidates.T >= 1).all(axis=0)
        costs = (candidates * (candidates @ G)).sum(axis=-1)
        feasible = costs >= 0
        feasible_candidates = candidates[separable & feasible]
        feasible_costs = costs[separable & feasible]

        # get the solution with minimal cost
        if len(feasible_candidates) > 0:
            w = feasible_candidates[np.argmin(feasible_costs)]
        else:
            w = candidates[1]  # the rank1 solution
        return w


class HyperbolicSVMHardSOS(SVM):
    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose=False,
        max_kappa=10,
        *kargs,
        **kwargs,
    ):
        """
        testing implementations with SumOfSquares
        :param max_kappa: maximum relaxation order to take (throw exception afterwards)
        """
        # ! python SumOfSquares is too slow !

        # get problem size
        dim = X.shape[1]

        # formulate problem
        w_symbolic = sp.symbols([f"w{d}" for d in range(dim)])

        for kappa in range(2, max_kappa):
            prob = poly_opt_prob(
                w_symbolic,
                sum(
                    [
                        -w_symbolic[d] ** 2 if d == 0 else w_symbolic[d] ** 2
                        for d in range(dim)
                    ]
                )
                / 2,
                ineqs=[
                    sum(
                        [
                            (
                                X[n][d] * w_symbolic[d]
                                if d == 0
                                else -X[n][d] * w_symbolic[d]
                            )
                            for d in range(dim)
                        ]
                    )
                    * (y[n] * 2 - 1)
                    - 1
                    for n in range(X.shape[0])
                ]  # correct classification
                + [
                    sum(
                        [
                            -w_symbolic[d] ** 2 if d == 0 else w_symbolic[d] ** 2
                            for d in range(dim)
                        ]
                    )
                ],  # domain requirement
                deg=kappa,
            )

            try:
                # solve the problem
                prob.solve(solver="mosek")

                # solution found, exit loop
                break
            except:
                warnings.warn(f"relaxation order kappa = {kappa} failed, increase by 1")

        # TODO: how to retrieve w?
