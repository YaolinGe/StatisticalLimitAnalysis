"""
This module handles the signal generation using gaussian process
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np


class GaussianProcess:

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gaussian_process.fit(self.x, self.y)
        print("Fitted covariance function: ", self.gaussian_process.kernel_)

    def get_realized_sample(self, x, num_realizations) -> np.ndarray:
        return self.gaussian_process.sample_y(x, num_realizations)



