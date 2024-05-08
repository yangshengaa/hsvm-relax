"""
SVM base class, handling binary and multiclass

for multi-class classification, we use OVR (one vs rest) strategy
"""

# load packages
import os
from typing import Dict
import pickle
from itertools import combinations
import numpy as np

# load platt
from .utils import platt_decision, get_platt_scaling_coef, objective_soft


class SVM:
    def __init__(self, multi_class="ovr", *kargs, **kwargs):
        """
        :param multi_class: "ovr" for one-vs-rest, "ovo" for one-vs-one
        """
        self._is_binary = None  # indicator for binary classification
        self._params = {}  # organize parameters in a dictionary
        self._obj = {}  # store objective values
        self.multi_class = multi_class

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=False, *kargs, **kwargs):
        """
        train the model (one-vs-rest)
        :param X: input data
        :param y: input label (0, 1, 2, ..., K)
        :param verbose: True to print training/optimization steps
        """
        # check if binary
        unique_labels = np.unique(y)

        # binary
        if len(unique_labels) == 2:
            self._is_binary = True
            y_binarized = (2 * y) - 1  # convert to -1 and 1
            self.fit_binary(X, y_binarized, verbose=verbose, *kargs, **kwargs)

        else:
            self._is_binary = False
            self._num_classes = len(unique_labels)

            # duck-type a dictionary to store platt coefficients
            self.platt_coefs_by_class = {}

            # for each class, fit a model
            if self.multi_class == "ovr":
                for k in range(self._num_classes):
                    y_binarized = (y == k) * 2 - 1
                    self.fit_binary(
                        X, y_binarized, verbose=verbose, k=k, *kargs, **kwargs
                    )

                    # platt training
                    decision_vals = self.decision_function(X, k=k)
                    a, b = get_platt_scaling_coef(
                        decision_vals,
                        y_binarized,
                        prior0=None,
                        prior1=None,
                        max_iteration=100,
                    )

                    # append to dict
                    self.platt_coefs_by_class[k] = (a, b)

            # for each pair of class, fit a model (ovo)
            elif self.multi_class == "ovo":
                for k1, k2 in combinations(range(self._num_classes), 2):
                    # select samples of these classes
                    selector = (y == k1) | (y == k2)
                    X_selected = X[selector]
                    y_selected = y[selector]
                    y_binarized = (y_selected == k1) * 2 - 1
                    self.fit_binary(
                        X_selected,
                        y_binarized,
                        verbose=verbose,
                        k=(k1, k2),
                        *kargs,
                        **kwargs,
                    )

                    # platt scaling
                    decision_vals = self.decision_function(X_selected, k=(k1, k2))
                    a, b = get_platt_scaling_coef(
                        decision_vals,
                        y_binarized,
                        prior0=None,
                        prior1=None,
                        max_iteration=100,
                    )

                    # append to dict
                    self.platt_coefs_by_class[(k1, k2)] = (a, b)

            else:
                raise NotImplementedError(
                    f"multi_class {self.multi_class} not supported"
                )

    def predict(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        """
        make prediction

        :param X: the test data
        """
        assert self._is_binary is not None, "the model is not fitted. Call .fit() first"
        if self._is_binary:
            return self._predict_binary(X, *kargs, **kwargs)
        else:
            if self.multi_class == "ovr":
                return self._predict_multi_ovr(X, *kargs, **kwargs)
            elif self.multi_class == "ovo":
                return self._predict_multi_ovo(X, *kargs, **kwargs)

    def _predict_binary(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        decision_vals = self.decision_function(X)
        decisions = (decision_vals >= 0).astype(int)
        return decisions

    def _predict_multi_ovr(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        """make prediction based on platt-scaling (One-vs-Rest)"""
        # for each class, get probability of belonging to this class
        decision_probabilities = []
        for k in range(self._num_classes):
            a, b = self.platt_coefs_by_class[k]
            decision_vals = self.decision_function(X, k)
            cur_prob = platt_decision(decision_vals, a, b)

            decision_probabilities.append(cur_prob)
        decision_probabilities = np.vstack(decision_probabilities).T

        # use argmax as the final decision
        decisions = np.argmax(decision_probabilities, axis=1)
        return decisions

    def _predict_multi_ovo(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        """make prediction based on platt-scaling (One-vs-One)"""
        # for each class pairs, get the decision using majority vote
        num_votes = np.zeros((X.shape[0], self._num_classes), dtype=int)
        for k1, k2 in combinations(range(self._num_classes), 2):
            a, b = self.platt_coefs_by_class[(k1, k2)]
            decision_vals = self.decision_function(X, (k1, k2))
            cur_prob = platt_decision(decision_vals, a, b)

            # convert probabilities to index to increment
            col_idx = np.where(cur_prob > 1 / 2, k1, k2)
            row_idx = np.arange(X.shape[0])
            num_votes[(row_idx, col_idx)] += 1
        # use argmax as the final decision
        decisions = np.argmax(num_votes, axis=1)
        return decisions

    def save(self, path: str, tag: str):
        """save a trained model"""
        # only need to save `_params`
        with open(os.path.join(path, f"model_params_{tag}.pkl"), "wb") as f:
            pickle.dump(self._params, f)

    def load(self, path: str, tag: str):
        """load a trained model"""
        with open(os.path.join(path, f"model_params_{tag}.pkl"), "rb") as f:
            self._params = pickle.load(f)

    def get_train_loss(self) -> Dict[int, float]:
        """get the train loss by class"""
        # change key to string for dumpping
        if self.multi_class == "ovo":
            temp = {}
            for key, val in self._obj.items():
                temp[str(key)] = val
            return temp
        else:
            return self._obj

    def get_test_loss(self, X: np.ndarray, y: np.ndarray, C: float) -> Dict[int, float]:
        """get the test loss by class"""
        test_loss = {}

        # binary
        if len(self._params) == 1:
            y_binarized = (2 * y) - 1  # convert to -1 and 1
            # test_loss[0] = objective_soft()
            w = self.get_predictor(0)
            test_loss[0] = objective_soft(w, X, y_binarized, C)
        else:
            if self.multi_class == "ovr":
                for k in range(len(self._params)):
                    y_binarized = (y == k) * 2 - 1
                    w = self.get_predictor(k)
                    test_loss[k] = objective_soft(w, X, y_binarized, C)
            elif self.multi_class == "ovo":
                for k1, k2 in combinations(list(range(self._num_classes)), 2):
                    # sample data of these two classes
                    selector = (y == k1) | (y == k2)
                    X_selected = X[selector]
                    y_selected = y[selector]
                    y_binarized = (y_selected == k1) * 2 - 1
                    w = self.get_predictor((k1, k2))
                    test_loss[f"({k1}, {k2})"] = objective_soft(
                        w, X_selected, y_binarized, C
                    )

        return test_loss

    def get_optimality_gap(self) -> Dict[int, float]:
        """get optimality gap"""
        if self.multi_class == "ovo":
            temp = {}
            for key, val in self._gaps.items():
                temp[str(key)] = val
            return temp
        else:
            return self._gaps

    # -----------------------------------------------
    # ========== custom implementations =============
    # -----------------------------------------------
    def fit_binary(
        self, X: np.ndarray, y: np.ndarray, verbose=False, k: int = 0, *kargs, **kwargs
    ):
        """
        binary fitting loop

        :param k: the current class index
        """
        raise NotImplementedError()

    def decision_function(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        """
        the raw decision values after fitting (same behavior as in sklearn)
        :param k: the class index
        """
        raise NotImplementedError()

    def get_predictor(self, k: int = 0) -> np.ndarray:
        """get the predictor"""
        raise NotImplementedError()
