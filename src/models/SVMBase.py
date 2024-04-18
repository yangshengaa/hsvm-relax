"""
SVM base class, handling binary and multiclass

for multi-class classification, we use OVR (one vs rest) strategy
"""

# load packages
import os
from typing import Dict
import pickle
import numpy as np

# load platt
from .utils import platt_decision, get_platt_scaling_coef, objective_soft


class SVM:
    def __init__(self, *kargs, **kwargs):
        self._is_binary = None  # indicator for binary classification
        self._params = {}  # organize parameters in a dictionary
        self._obj = {}  # store objective values

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=False, *kargs, **kwargs):
        """
        train the model
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

        # multiclass (OVR)
        else:
            self._is_binary = False
            self._num_classes = len(unique_labels)

            # duck-type a dictionary to store platt coefficients
            self.platt_coefs_by_class = {}

            # for each class, fit a model
            for k in range(self._num_classes):
                y_binarized = (y == k) * 2 - 1
                self.fit_binary(X, y_binarized, verbose=verbose, k=k, *kargs, **kwargs)

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

    def predict(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        """
        make prediction

        :param X: the test data
        """
        assert self._is_binary is not None, "the model is not fitted. Call .fit() first"
        if self._is_binary:
            return self._predict_binary(X, *kargs, **kwargs)
        else:
            return self._predict_multi(X, *kargs, **kwargs)

    def _predict_binary(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        decision_vals = self.decision_function(X)
        decisions = (decision_vals >= 0).astype(int)
        return decisions

    def _predict_multi(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        """make prediction based on platt-scaling"""
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
            for k in range(len(self._params)):
                y_binarized = (y == k) * 2 - 1
                w = self.get_predictor(k)
                test_loss[k] = objective_soft(w, X, y_binarized, C)
        return test_loss

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
