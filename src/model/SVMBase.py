"""
SVM base class, handling binary and multiclass

for multi-class classification, we use OVR (one vs rest) strategy
"""

# load packages
import numpy as np


class SVM:
    def __init__(self, *kargs, **kwargs):
        self._is_binary = None

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
            for k in range(len(unique_labels)):
                y_binarized = (y == k) * 2 - 1
                self.fit_binary(X, y_binarized, verbose=verbose, *kargs, **kwargs)

                # TODO: handle properly the multiclass fit function

    def predict(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        """
        make prediction

        :param X: the test data
        """
        assert self._is_binary is not None, "the model is not fitted. Call .fit() first"
        if self._is_binary:
            return self.predict_binary(X, *kargs, **kwargs)
        else:
            return self.predict_multi(X, *kargs, **kwargs)

    # -----------------------------------------------
    # ========== custom implementations =============
    # -----------------------------------------------
    def fit_binary(self, X: np.ndarray, y: np.ndarray, verbose=False, *kargs, **kwargs):
        raise NotImplementedError()

    def predict_binary(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def predict_multi(self, X: np.ndarray, *kargs, **kwargs) -> np.ndarray:
        raise NotImplementedError()


# TODO: Platt Scaling for multiclass classification
