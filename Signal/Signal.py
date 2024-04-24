"""
Signal class handles the signal generation and processing for the statistical limit analysis
"""
import pandas as pd
import numpy as np
from Signal.GaussianProcess import GaussianProcess


class Signal:

    def __init__(self, datapath: str = "./sampledata/Load_demo.csv") -> None:
        self.datapath = datapath
        self.data = pd.read_csv(self.datapath).to_numpy()
        self.x = self.data[:, 0].reshape(-1, 1)
        self.y = self.data[:, 1].reshape(-1, 1)
        ind_train = np.random.choice(np.arange(self.y.size), size=6, replace=False)
        self.x_train, self.y_train = self.x[ind_train], self.y[ind_train]
        self.gp = GaussianProcess(self.x_train, self.y_train)

    def generate_random_signal(self):
        return self.gp.get_realized_sample(self.x, 1)

    def generate_signal_using_gaussian_process(self) -> np.ndarray:

        pass

    def setup_gaussian_process(self):
        pass





if __name__ == "__main__":
    pass
