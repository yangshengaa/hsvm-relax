"""
Soft SVM
- Euclidean SVM
- Hyperbolic SVM (original problem)
- Hyperbolic SVM (SDP relaxation)
- Hyperbolic SVM Dual Formulation (SOS relaxation)
- Hyperbolic SVM Primal Formulation (Moment Relaxation)
- Hyperbolic SVM Sparse Primal (exploit star-shape RIP condition, the go-to method)

Empirically Primal is faster than Dual
"""

# load packages
import warnings
from math import comb
from typing import List, Tuple
import numpy as np
import mosek
import scipy
import sympy as sp
from sympy.core.symbol import Symbol
from SumOfSquares import poly_opt_prob

from .SVMBase import SVM
from .utils import (
    stream_printer,
    check_solution_status,
    parse_sparse_linear_constraints,
    minkowski_product,
    monomials,
    svec2smat_batch,
)

# for symbolic placeholders
_INF = 0.0
_EPS = 1e-6


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

            primal_obj = task.getprimalobj(mosek.soltype.itr)

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
            solution_value = (-w_[0] ** 2 + sum(w_[1:] ** 2)) / 2 + np.clip(
                np.arcsinh(1) - np.arcsinh(B @ w_), a_min=0.0, a_max=None
            ).sum() * self.C
            self._params[k] = [W_, z_, w_]

            # get optimality gap
            eta = (solution_value - primal_obj) / (1 + solution_value + primal_obj)

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print("Optimal Solution: ")
                print("W: \n", self._params[k][0])
                print("z: \n", self._params[k][1])
                print("w: \n", self._params[k][2])
                print(f"Optimal Value: {solution_value:.4f}")
                print(f"Optimality gap: {eta:.4f}")

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
        epochs: int = 2000,
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
            grad_w = self._grad_fn(w_new, X, y, self.C)
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


class HyperbolicSVMSoftSOSDual(SVM):
    def __init__(self, C: float = 1.0, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.C = C

    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose=False,
        k: int = 0,
        max_kappa=10,
        *kargs,
        **kwargs,
    ):
        # get problem size
        n, dim = X.shape

        # formulate problem
        w = sp.symbols(f"w0:{dim}")
        xi = sp.symbols(f"xi1:{n+1}")
        decision_vars = (*w, *xi)

        obj = (-w[0] ** 2 + sum(w[i] ** 2 for i in range(1, dim))) / 2 + self.C * sum(
            xi
        )
        # add inequality constraints
        inequalities = []
        for i in range(n):
            cur_inequality = (
                y[i] * (X[i][0] * w[0] - sum(X[i][k] * w[k] for k in range(1, dim)))
                + np.sqrt(2) * xi[i]
                - 1
            )
            inequalities.append(cur_inequality)
            inequalities.append(xi[i])
        inequalities.append(-w[0] ** 2 + sum(w[k] ** 2 for k in range(1, dim)))
        # no equality constraints
        equalities = []

        # formulate problem
        for kappa in range(3, max_kappa):
            prob = poly_opt_prob(
                decision_vars, obj, eqs=equalities, ineqs=inequalities, deg=kappa
            )
            try:
                # solve the problem
                prob.solve(solver="mosek")

                # solution found, exit loop
                print(prob.value)
                break
            except:
                warnings.warn(f"relaxation order kappa = {kappa} failed, increase by 1")


class HyperbolicSVMSoftSOSPrimal(SVM):
    """
    implement the primal moment relaxation

    throughout this implementation, we use w to denote the decision boundary
    and xi as the slack variable

    note that MOSEK by default SVEC a matrix by column. The order is going through
    one column before going to the next column
    """

    def __init__(self, C: float = 1.0, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.C = C

    def _get_moment_matrix_svec(
        self, basis: List[Symbol], y_symbolic: List[Symbol]
    ) -> List[List[float]]:
        """
        parse moment matrix constraint
        """
        monomial_idx_map = dict(zip(y_symbolic, range(len(y_symbolic))))
        num_rows = int(len(basis) * (len(basis) + 1) / 2)
        # moment_matrix_svec = np.zeros((num_rows, len(y_symbolic)))

        val_list, row_list, col_list = [], [], []

        # populate moment matrix (svec)
        row_idx = 0
        for j in range(len(basis)):
            for i in range(j, len(basis)):
                cur_monomial = basis[i] * basis[j]
                cur_idx = monomial_idx_map[cur_monomial]
                # moment_matrix_svec[row_idx][cur_idx] = 1 if (i == j) else np.sqrt(2)

                val_list.append(1 if (i == j) else np.sqrt(2))
                row_list.append(row_idx)
                col_list.append(cur_idx)
                row_idx += 1

        # return moment_matrix_svec
        return val_list, row_list, col_list

    def _get_localizing_matrix_svec(
        self,
        kappa: int,
        decision_vars: List[Symbol],
        y_symbolic,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[np.ndarray]:
        """
        parse localizing matrix constraint
        :return list of localizing matrix constraint svec-ed
        """
        # prepare data
        N, dim = X.shape
        G_prime = -np.eye(dim)
        G_prime[0, 0] = 1
        B = y.reshape(-1, 1) * (X @ G_prime)

        monomial_idx_map = dict(zip(y_symbolic, range(len(y_symbolic))))

        # in this problem, all constraints have max degree 2
        # so the localizing matrix has basis kappa - 1
        basis_con = monomials(decision_vars, range(kappa))
        Sn_kappa = comb(len(decision_vars) + kappa, kappa)
        offset_idx = int(Sn_kappa * (1 + Sn_kappa) / 2)

        # cache basis_con outer products
        basis_monomial_idx_map = {}
        for j in range(len(basis_con)):
            for i in range(j, len(basis_con)):
                basis_monomial_idx_map[(i, j)] = basis_con[i] * basis_con[j]

        num_rows = int(len(basis_con) * (len(basis_con) + 1) / 2)

        val_list, row_list, col_list = [], [], []

        for n in range(N):
            row_idx = 0
            for j in range(len(basis_con)):
                for i in range(j, len(basis_con)):
                    cached = basis_monomial_idx_map[(i, j)]
                    cur_row_idx = row_idx + offset_idx + num_rows * n * 2

                    # xi
                    xi_n = decision_vars[dim + n]
                    idx = monomial_idx_map[xi_n * cached]
                    val_list.append(1 if (i == j) else np.sqrt(2))
                    row_list.append(cur_row_idx)
                    col_list.append(idx)

                    # cls
                    # xi
                    val_list.append(np.sqrt(2) if (i == j) else 2)
                    row_list.append(cur_row_idx + num_rows)
                    col_list.append(idx)

                    # w0, w1, ...
                    for d in range(dim):
                        wd = decision_vars[d]
                        idx = monomial_idx_map[cached * wd]
                        val = B[n][d]

                        val_list.append(val if (i == j) else np.sqrt(2) * val)
                        col_list.append(idx)
                        row_list.append(cur_row_idx + num_rows)

                    # -1
                    val_list.append(-1 if (i == j) else -np.sqrt(2))
                    row_list.append(cur_row_idx + num_rows)
                    col_list.append(monomial_idx_map[cached])

                    row_idx += 1

        # w^T G w >= 0
        row_idx = 0
        for j in range(len(basis_con)):
            for i in range(j, len(basis_con)):
                cached = basis_monomial_idx_map[(i, j)]
                cur_row_idx = row_idx + offset_idx + 2 * N * num_rows
                for d in range(dim):
                    wd = decision_vars[d]
                    cur_term = wd**2
                    idx = monomial_idx_map[cur_term * cached]
                    if d == 0:
                        val_list.append(-1 if (i == j) else -np.sqrt(2))
                        col_list.append(idx)
                        row_list.append(cur_row_idx)

                    else:
                        val_list.append(1 if (i == j) else np.sqrt(2))
                        col_list.append(idx)
                        row_list.append(cur_row_idx)

                row_idx += 1

        return val_list, row_list, col_list

    def _get_cj(self, w: List[Symbol], xi: List[Symbol], y_symbolic: List[Symbol]):
        """get objective value and position"""
        c_idx = []
        c_val = []

        # populate w
        for d, wd in enumerate(w):
            c_idx.append(y_symbolic.index(wd**2))
            c_val.append(1 / 2 if (d > 0) else -1 / 2)

        # populate xi
        for xi_n in xi:
            c_idx.append(y_symbolic.index(xi_n))
            c_val.append(self.C)

        return c_idx, c_val

    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose=False,
        k: int = 0,
        kappa=3,
        *kargs,
        **kwargs,
    ):
        n, dim = X.shape
        w = sp.symbols(f"w0:{dim}")
        xi = sp.symbols(f"xi1:{n + 1}")
        decision_vars = (*w, *xi)

        G = np.eye(dim)
        G[0, 0] = -1.0

        # only specify lower triangular part, divide by 2
        B = (X @ -G) * y.reshape(-1, 1)

        # formulate problem
        Sn_kappa1 = comb(
            len(decision_vars) + kappa - 1, kappa - 1
        )  # shape of localizing matrix
        Sn_kappa = comb(len(decision_vars) + kappa, kappa)  # shape of moment matrix

        # compute length of moment and localizing matrices
        moment_matrix_shape = Sn_kappa * (Sn_kappa + 1) // 2
        localizing_matrix_shape = Sn_kappa1 * (Sn_kappa1 + 1) // 2

        # y \in S(n, 2kappa)
        y_symbolic = monomials(decision_vars, range(2 * kappa + 1))

        # create the standard monomials
        basis = monomials(decision_vars, range(kappa + 1))
        moment_vals, moment_rows, moment_cols = self._get_moment_matrix_svec(
            basis, y_symbolic
        )
        local_vals, local_rows, local_cols = self._get_localizing_matrix_svec(
            kappa, decision_vars, y_symbolic, X, y
        )

        f_entries = moment_vals + local_vals
        f_cols = moment_cols + local_cols
        f_rows = moment_rows + local_rows
        total_afe_expr = moment_matrix_shape + localizing_matrix_shape * (2 * n + 1)

        # get obj
        c_idx, c_val = self._get_cj(w, xi, y_symbolic)

        num_vars = len(y_symbolic)
        with mosek.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, stream_printer)

            # add y
            bkx = [mosek.boundkey.fx] + [mosek.boundkey.fr] * (num_vars - 1)
            blx = [1.0] + [-_INF] * (num_vars - 1)
            bux = [1.0] + [+_INF] * (num_vars - 1)
            task.appendvars(num_vars)
            task.putvarboundlist(range(num_vars), bkx, blx, bux)

            # add svec cones
            task.appendafes(total_afe_expr)
            task.putafefentrylist(f_rows, f_cols, f_entries)
            task.appendacc(
                task.appendsvecpsdconedomain(moment_matrix_shape),
                range(moment_matrix_shape),
                None,
            )
            for k in range(2 * n + 1):
                task.appendacc(
                    task.appendsvecpsdconedomain(localizing_matrix_shape),
                    range(
                        moment_matrix_shape + localizing_matrix_shape * k,
                        moment_matrix_shape + localizing_matrix_shape * (k + 1),
                    ),
                    None,
                )

            # add objective
            task.putclist(c_idx, c_val)
            task.putobjsense(mosek.objsense.minimize)

            # run optimizer
            task.optimize()

            # check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            check_solution_status(solsta)

            primal_obj = task.getprimalobj(mosek.soltype.itr)

            # get y and store solution
            # y_sol = np.array(task.getxx(mosek.soltype.itr))
            # w_and_xi = self._get_optima(
            #     kappa,
            #     decision_vars,
            #     basis,
            #     y_sol,
            #     moment_matrix_shape,
            #     moment_vals,
            #     moment_rows,
            #     moment_cols,
            # )
            # w_ = w_and_xi[:dim]
            w_ = np.array(task.getxxslice(mosek.soltype.itr, 1, 1 + dim))
            self._params[k] = w_
            solution_value = (-w_[0] ** 2 + sum(w_[1:] ** 2)) / 2 + np.clip(
                np.arcsinh(1) - np.arcsinh(B @ w_), a_min=0.0, a_max=None
            ).sum() * self.C

            # get optimality gap
            eta = (solution_value - primal_obj) / (1 + primal_obj + solution_value)

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print(f"w: {w_}")
                print(f"solution value: {solution_value:.4f}")
                print(f"optimality gap: {eta:.4f}")

    # def _get_optima(
    #     self,
    #     kappa: int,
    #     decision_vars: List[Symbol],
    #     basis: List[Symbol],
    #     y_sol: np.ndarray,
    #     moment_length: int,
    #     moment_vals: List[float],
    #     moment_rows: List[int],
    #     moment_cols: List[int],
    # ):
    #     """
    #     parse solution using flat truncation theory
    #     :param kappa: the relaxation order
    #     :param decision_vars: the decision values
    #     :param basis: the basis generated by decision_vars of degree kappa
    #     :param y_sol: the solution from mosek
    #     :param moment_length: use to transform sparse matrix to dense
    #     :param moment_val: the value in the moment matrix multipliers
    #     :param moment_rows, moment_cols: the positions for the moment matrix multiplier
    #     """
    #     # * temporary solution: taking the nominal values, already good enough
    #     solution = y_sol[1 : 1 + len(decision_vars)]
    #     return solution

    # # from sprase to dense
    # moment_mat_svec = coo_array(
    #     (moment_vals, (moment_rows, moment_cols)), shape=(moment_length, len(y_sol))
    # ).toarray()

    # # get original matrix
    # mat_batch = svec2smat_batch(moment_mat_svec)
    # moment_matrix = (mat_batch * y_sol.reshape(-1, 1, 1)).sum(axis=0)

    # # # in this problem, we have d_c = d_0 = 1
    # # idx = None
    # # for k in range(kappa, 0, -1):
    # #     num_large, num_small = comb(num_vars + k, k), comb(num_vars + k - 1, k - 1)
    # #     rank_large = np.linalg.matrix_rank(
    # #         moment_matrix[:num_large][:, :num_large], _EPS, hermitian=True
    # #     )
    # #     rank_small = np.linalg.matrix_rank(
    # #         moment_matrix[:num_small][:, :num_small], _EPS, hermitian=True
    # #     )
    # #     if  rank_large == rank_small:
    # #         idx = k
    # #         break

    # # get the slice from the solution
    # num_vars = len(decision_vars)

    # # * take critical component (less effective than directly taking the solution)
    # # print(moment_matrix[:1+num_vars][:, :1+num_vars])
    # # print(np.linalg.eigvalsh(moment_matrix[:1+num_vars][:, :1+num_vars]))
    # # critical_component = moment_matrix[:1+num_vars][:, :1+num_vars]
    # # print(critical_component)

    # # # assert rank-1 of this matrix
    # # critical_eigvals, critical_eigvecs = np.linalg.eigh(critical_component)
    # # assert sum(critical_eigvals >= _EPS) == 1, f"rank of matrix exceed 1, with spctrum {critical_eigvals}"
    # # # take top 1 component
    # # solution = (critical_eigvecs[:, -1] * critical_eigvals[-1])[1:]
    # solution = moment_matrix[0, 1:num_vars+1]
    # return solution

    # # * flat extension, which is not used, since this problem does not
    # * satisfy flat extensionm it looks like
    # eigvals, eigvecs = np.linalg.eigh(moment_matrix)
    # temp = moment_matrix[1:6][:, 1:6]
    # print(temp)
    # print(np.linalg.eigvalsh(temp))
    # V = eigvecs[:, -(num_vars + 1) :] * np.sqrt(
    #     eigvals[-(num_vars + 1) :].reshape(1, -1)
    # )  # by default ascending
    # U, pivots = sp.Matrix(V.T).rref()
    # assert set(pivots) == set(list(range(num_vars + 1)))
    # U = np.array(U).astype(np.float64).T
    # numerically suppress small values
    # U[U < _EPS] = 0

    # # find companion matrix
    # basis_idx_map = dict(zip(basis, range(len(basis))))
    # companion_matrices = []
    # for i, var in enumerate(decision_vars):
    #     idx_list = [i + 1]
    #     for var2 in decision_vars:
    #         idx_list.append(basis_idx_map[var * var2])
    #     companion_matrices.append(U[idx_list])

    # # robust method of solution extraction, using schur decomposition
    # # we pick generic combinations with weights one
    # M = sum(companion_matrices)
    # T, U = schur(M, output="complex")
    # print(T)
    # for i, companion_matrix in enumerate(companion_matrices):
    #     print(U.conj().T @ companion_matrix @ U)
    #     pass


class HyperbolicSVMSoftSOSSparsePrimal(SVM):
    def __init__(self, C: float = 1.0, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.C = C

    def _get_sparse_binding(
        self, g0_y: List[Symbol], gi_y: List[Symbol], N: int
    ) -> Tuple[List[float], List[int], List[int]]:
        """
        get the overlap parts of group 0 and group i to be the same
        :param g0_y: the decision variables of degree 2 * kappa generated by group 0
        :param gi_y: the decision variables of degree 2 * kappa generated by group i
        :param N: number of group i, equaling the number of samples in data
        :return the triplet sparse representation of binding constraints
        """
        g0_ylen = len(g0_y)
        gi_ylen = len(gi_y)
        gi_var_map = dict(zip(gi_y, range(gi_ylen)))
        binding_idx_list = []
        for k in range(1, g0_ylen):  # exclude binding the leading 1
            var = g0_y[k]
            binding_idx_list.append(gi_var_map[var])
        binding_idx_arr = np.array(binding_idx_list)

        A_rows = list(range(len(binding_idx_list) * (N - 1))) * 2
        A_cols = (
            binding_idx_list * (N - 1)
            + (binding_idx_arr + (np.arange(1, N) * gi_ylen).reshape(-1, 1))
            .flatten()
            .tolist()
        )
        A_vals = [1.0] * len(binding_idx_list) * (N - 1) + [-1.0] * len(
            binding_idx_list
        ) * (N - 1)

        return A_vals, A_rows, A_cols

    def _get_cj(
        self, gi_y: List[Symbol], N: int, w: List[Symbol]
    ) -> Tuple[List[float], List[int]]:
        """
        get objective
        :param gi_y: the decision variables of degree 2 * kappa generated by group i
        :param N: number of group i, equaling the number of samples in data
        :param w: the list of symbols for decision boundaries
        :return cj
        """
        # get 1/2 w^T G w from the first group
        c_val, c_idx = [], []
        gi_ylen = len(gi_y)
        gi_var_map = dict(zip(gi_y, range(gi_ylen)))
        for d, var in enumerate(w):
            idx = gi_var_map[var**2]
            c_idx.append(idx)
            c_val.append(1 / 2 if (d > 0) else -1 / 2)

        # get xi from other groups
        # we know exactly where xi is
        unshifted_idx = len(w) + 1
        c_idx += (unshifted_idx + np.arange(N) * gi_ylen).tolist()
        c_val += [self.C] * N

        return c_val, c_idx

    def _get_moment_constraints(
        self,
        gi_basis: List[Symbol],
        gi_basis_con: List[Symbol],
        gi_y: List[Symbol],
        N: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[List[float], List[int], List[int]]:
        """
        parse moment constraints into svec conic constraints
        :param gi_basis: the basis of degree kappa
        :param gi_basis_con: the basis of degree kappa - 1 (for constraints)
        :param gi_y: the basis of degree 2 * kappa
        :return the sparse triplet representations of affine conic expressions
        """
        # prepare data
        N, dim = X.shape
        G_prime = -np.eye(dim)
        G_prime[0, 0] = 1
        B = y.reshape(-1, 1) * (X @ G_prime)

        cols, vals = [], []
        # 1. moment matrix
        # parse a single moment matrix (svec)
        var_idx_map = dict(zip(gi_y, range(len(gi_y))))
        for j in range(len(gi_basis)):
            for i in range(j, len(gi_basis)):
                cur_monomial = gi_basis[i] * gi_basis[j]
                cur_idx = var_idx_map[cur_monomial]

                vals.append(1 if (i == j) else np.sqrt(2))
                cols.append(cur_idx)
        # repeat n times
        cols = (
            (np.array(cols) + np.arange(N).reshape(-1, 1) * len(gi_y))
            .flatten()
            .tolist()
        )
        vals = vals * N
        rows = list(range(len(vals)))

        # 2. localizing matrix (svec)
        vals_local, cols_local, rows_local = [], [], []
        offset_idx = len(rows)
        num_rows = len(gi_basis_con) * (len(gi_basis_con) + 1) // 2

        xi_symbol = gi_basis[dim + 1]
        for n in range(N):
            row_idx = 0
            for j in range(len(gi_basis_con)):
                for i in range(j, len(gi_basis_con)):
                    cur_monomial = gi_basis_con[i] * gi_basis_con[j]
                    cur_row_idx = row_idx + offset_idx + num_rows * n * 2

                    # xi
                    col_idx = var_idx_map[cur_monomial * xi_symbol] + n * len(gi_y)
                    cols_local.append(col_idx)
                    rows_local.append(cur_row_idx)
                    vals_local.append(1 if (i == j) else np.sqrt(2))

                    # classification constraints
                    # xi
                    cols_local.append(col_idx)
                    rows_local.append(cur_row_idx + num_rows)
                    vals_local.append(np.sqrt(2) if (i == j) else 2)

                    # w0, w1, ...
                    for d in range(dim):
                        wd = gi_basis[d + 1]
                        col_idx = var_idx_map[cur_monomial * wd] + n * len(gi_y)
                        cur_val = B[n][d]

                        vals_local.append(cur_val if (i == j) else np.sqrt(2) * cur_val)
                        cols_local.append(col_idx)
                        rows_local.append(cur_row_idx + num_rows)

                    # -1
                    vals_local.append(-1 if (i == j) else -np.sqrt(2))
                    rows_local.append(cur_row_idx + num_rows)
                    cols_local.append(var_idx_map[cur_monomial] + n * len(gi_y))

                    row_idx += 1

        # w^T G w >= 0, asign to the first group
        row_idx = 0
        for j in range(len(gi_basis_con)):
            for i in range(j, len(gi_basis_con)):
                cur_monomial = gi_basis_con[i] * gi_basis_con[j]
                cur_row_idx = row_idx + offset_idx + 2 * N * num_rows
                for d in range(dim):
                    wd = gi_basis[d + 1]
                    cur_term = wd**2
                    idx = var_idx_map[cur_term * cur_monomial]
                    if d == 0:
                        vals_local.append(-1 if (i == j) else -np.sqrt(2))
                        cols_local.append(idx)
                        rows_local.append(cur_row_idx)

                    else:
                        vals_local.append(1 if (i == j) else np.sqrt(2))
                        cols_local.append(idx)
                        rows_local.append(cur_row_idx)

                row_idx += 1

        afe_vals = vals + vals_local
        afe_cols = cols + cols_local
        afe_rows = rows + rows_local
        return afe_vals, afe_rows, afe_cols

    def fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose=False,
        k: int = 0,
        kappa=3,
        *kargs,
        **kwargs,
    ):
        n, dim = X.shape
        w = sp.symbols(f"w0:{dim}")
        xi = sp.symbols("xi")

        G = np.eye(dim)
        G[0, 0] = -1.0

        # only specify lower triangular part, divide by 2
        B = (X @ -G) * y.reshape(-1, 1)

        # set decision variable
        # group 0 = (w_0, w_1, ..., w_d), as dummy variables for indexing, not modeled
        g0_y = monomials(w, range(2 * kappa + 1))
        g0_ylen = len(g0_y)

        # group i = (w_0, w_1, ..., w_d, \xi_i)
        gi_y = monomials([*w, xi], range(2 * kappa + 1))
        gi_ylen = len(gi_y)
        gi_basis = monomials([*w, xi], range(kappa + 1))
        gi_basis_con = monomials([*w, xi], range(kappa))

        # number of variables
        num_vars = n * gi_ylen

        # get binding constraints
        A_vals, A_rows, A_cols = self._get_sparse_binding(g0_y, gi_y, n)
        num_cons = (g0_ylen - 1) * (n - 1)

        # get objective
        c_val, c_idx = self._get_cj(gi_y, n, w)

        # get moment matrix
        afe_vals, afe_rows, afe_cols = self._get_moment_constraints(
            gi_basis, gi_basis_con, gi_y, n, X, y
        )
        total_afe_expr = afe_rows[-1] + 1
        moment_con_dim = len(gi_basis) * (len(gi_basis) + 1) // 2
        local_con_dim = len(gi_basis_con) * (len(gi_basis_con) + 1) // 2

        with mosek.Task() as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, stream_printer)

            # add variables
            bkx = ([mosek.boundkey.fx] + [mosek.boundkey.fr] * (gi_ylen - 1)) * n
            blx = ([1.0] + [-_INF] * (gi_ylen - 1)) * n
            bux = ([1.0] + [_INF] * (gi_ylen - 1)) * n

            task.appendvars(num_vars)
            task.putvarboundlist(range(num_vars), bkx, blx, bux)

            # add sparse binding
            bkc = [mosek.boundkey.fx] * num_cons
            blc = [0.0] * num_cons
            buc = [0.0] * num_cons

            task.appendcons(num_cons)
            task.putaijlist(A_rows, A_cols, A_vals)
            task.putconboundlist(range(num_cons), bkc, blc, buc)

            # add moment constraints
            task.appendafes(total_afe_expr)
            task.putafefentrylist(afe_rows, afe_cols, afe_vals)
            for i in range(n):
                task.appendacc(
                    task.appendsvecpsdconedomain(moment_con_dim),
                    range(moment_con_dim * i, moment_con_dim * (i + 1)),
                    None,
                )
            for i in range(2 * n + 1):
                task.appendacc(
                    task.appendsvecpsdconedomain(local_con_dim),
                    range(
                        moment_con_dim * n + local_con_dim * i,
                        moment_con_dim * n + local_con_dim * (i + 1),
                    ),
                    None,
                )

            # add objective
            task.putclist(c_idx, c_val)
            task.putobjsense(mosek.objsense.minimize)

            # run optimizer
            task.optimize()

            # check solution status
            solsta = task.getsolsta(mosek.soltype.itr)
            check_solution_status(solsta)

            primal_obj = task.getprimalobj(mosek.soltype.itr)

            # * get y and store solution (naive)
            w_ = np.array(task.getxxslice(mosek.soltype.itr, 1, 1 + dim))
            self._params[k] = w_
            solution_value = (-w_[0] ** 2 + sum(w_[1:] ** 2)) / 2 + np.clip(
                np.arcsinh(1) - np.arcsinh(B @ w_), a_min=0.0, a_max=None
            ).sum() * self.C

            # compute optimality gap
            eta = (solution_value - primal_obj) / (1 + primal_obj + solution_value)

            if verbose:
                task.solutionsummary(mosek.streamtype.msg)
                print(f"w: {w_}")
                print(f"solution value: {solution_value:.4f}")
                print(f"optimality gap: {eta:.4f}")

    def decision_function(self, X: np.ndarray, k: int = 0):
        w = self._params[k]
        decision_vals = minkowski_product(X, w)
        return decision_vals
