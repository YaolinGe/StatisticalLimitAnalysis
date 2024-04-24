"""
This test file will test the functions in the GaussianProcess.py file.
"""

from unittest import TestCase
from Signal.GaussianProcess import GaussianProcess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TestGaussianProcess(TestCase):

    def setUp(self):
        # self.data = pd.read_csv("./sampledata/Load.csv")
        # self.data['Time'] = pd.to_timedelta(self.data['Time']).dt.total_seconds()
        # self.dataset = self.data.to_numpy()
        # self.x = self.dataset[:, 0].reshape(-1, 1)
        # self.y = self.dataset[:, 1].reshape(-1, 1)

        self.x = np.linspace(0, 10, 100).reshape(-1, 1)
        # self.y = self.x * np.sin(self.x)
        self.y = self.x * np.sin(self.x) + np.random.normal(0, 0.1, self.x.shape)
        index_even = np.arange(self.y.size)[::10]
        index_train = index_even
        # index_train = np.random.choice(np.arange(self.y.size), size=6, replace=False)
        x_train, y_train = self.x[index_train], self.y[index_train]
        self.gp = GaussianProcess(x_train, y_train)

    def test_get_realized_sample(self):
        num_realizations = 10
        realizations = self.gp.get_realized_sample(self.x, num_realizations)
        plt.figure(figsize=(10, 5))
        for i in range(num_realizations):
            plt.plot(self.x, realizations[:, i], lw=1, linestyle='--')
        plt.scatter(self.gp.x, self.gp.y, color="red", label="Training points")
        plt.plot(self.x, self.y, label=r"$f(x) = x \ sin(x)$", linestyle="dotted")
        plt.show()
