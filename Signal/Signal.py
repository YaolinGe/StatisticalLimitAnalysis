"""
Signal class handles the signal generation and processing for the statistical limit analysis
"""
import pandas as pd
import numpy as np
import scipy
from statsmodels.tsa.arima.model import ARIMA
from hmmlearn import hmm
from pykalman import KalmanFilter
from Signal.GaussianProcess import GaussianProcess


class Signal:

    def __init__(self, datapath: str = "./sampledata/Load_demo.csv") -> None:
        self.datapath = datapath
        self.data = pd.read_csv(self.datapath).to_numpy()
        self.x = self.data[:, 0].reshape(-1, 1)
        self.y = self.data[:, 1].reshape(-1, 1)
        ind_train = np.random.choice(np.arange(self.y.size), size=10, replace=False)
        self.x_train, self.y_train = self.x[ind_train], self.y[ind_train]

    def generate_signal_replicate(self, num_realizations: int) -> np.ndarray:
        self.signals = np.empty((num_realizations, self.y.size))
        for i in range(num_realizations):
            time_shift = np.random.choice([-5, 5])
            shifted_signal = np.roll(self.y, time_shift, axis=0)
            noise = np.random.normal(0, 0.1, self.y.size) * np.amax(self.y)
            self.signals[i] = shifted_signal.flatten() + noise

    def generate_random_signal(self):
        return self.y + np.random.normal(0, 1, self.y.shape) * 0.025 * np.amax(self.y)

    def generate_signal_using_gaussian_process(self) -> np.ndarray:
        self.gp = GaussianProcess(self.x_train, self.y_train)
        return self.gp.get_realized_sample(self.x, 1)

    def generate_signal_using_arima(self) -> np.ndarray:
        # Fit an ARIMA model
        model = ARIMA(self.y_train, order=(1, 1, 1))  # Modify order as needed
        model_fit = model.fit()

        # Generate predictions
        prediction = model_fit.forecast(steps=self.x.size)
        return prediction

    def generate_signal_using_bootstrap(self) -> np.ndarray:
        # Using block bootstrap for time series
        block_size = 5  # Adjust block size based on the autocorrelation structure
        num_blocks = int(self.x_train.size / block_size)
        indices = np.random.choice(num_blocks, size=num_blocks, replace=True) * block_size
        sample_indices = np.array([i + np.arange(block_size) for i in indices]).flatten()
        sample_indices = sample_indices[sample_indices < self.x_train.size]

        # Create a new dataset by sampling blocks
        bootstrap_sample = self.y_train[sample_indices]
        return bootstrap_sample

    def generate_signal_using_hmm(self) -> np.ndarray:
        # Assuming the number of hidden states needs to be determined
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(self.y_train)

        # Generate data
        _, samples = model.sample(self.x.size)
        return samples

    def generate_signal_using_kalman_filter(self) -> np.ndarray:
        # Define the Kalman Filter
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=np.mean(self.y),
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)

        # Use the Kalman Filter to estimate the states
        state_means, _ = kf.filter(self.y)

        # Generate new time series data from the state estimates
        # Here, we simulate forward using the last state mean
        simulated_data, _ = kf.sample(n_timesteps=self.y.size, initial_state=state_means[0])
        return simulated_data.flatten()

    def get_confidence_interval(self, confidence_level: float) -> np.ndarray:
        """
        This function calculates the confidence interval for a given confidence level.

        Args:
            confidence_level (float): The desired confidence level (e.g., 0.95 for a 95% confidence interval).

        Returns:
            np.ndarray: The lower and upper bounds of the confidence interval.
        """
        # Calculate the mean prediction and standard deviation
        mean_prediction, std_prediction = self.gp.gaussian_process.predict(self.x, return_std=True)

        # Calculate the z-score for the desired confidence level
        z_score = scipy.stats.norm.ppf(confidence_level, loc=0, scale=1)

        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = mean_prediction - z_score * std_prediction
        upper_bound = mean_prediction + z_score * std_prediction

        return np.column_stack((lower_bound, upper_bound))


if __name__ == "__main__":
    pass
