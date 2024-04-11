import unittest
import numpy as np
from usr_func.replicate_signals import replicate_signals
import seaborn as sns
import matplotlib.pyplot as plt


class TestReplicateSignals(unittest.TestCase):
    def setUp(self):
        """ Generate a sine wave dataset for testing """
        t = np.linspace(0, 10, 100)
        self.dataset = np.array([t, np.sin(t)]).T

    def test_replicate_signals(self):
        result = replicate_signals(self.dataset, number_of_replicas=10, noise_level=.1, noise_seed=0, time_shift_range=2)
        """ Plot the original signal and the replicated signals and the mean of the replicated signals """
        plt.figure()
        plt.plot(self.dataset[:, 0], self.dataset[:, 1], label='Original Signal')

        for signal in result:
            plt.plot(self.dataset[:, 0], signal, color='gray', alpha=0.1)

        mean = np.mean(result, axis=0)
        plt.plot(self.dataset[:, 0], mean, color='red', label='Mean of Replicated Signals')

        std = np.std(result, axis=0, ddof=1)
        # Calculate the upper and lower bounds for the standard deviations
        upper_bound = mean + std
        lower_bound = mean - std

        # Plot the upper and lower bounds as shaded regions
        sns.lineplot(x=self.dataset[:, 0], y=upper_bound, color='blue', label='Upper Bound')
        sns.lineplot(x=self.dataset[:, 0], y=lower_bound, color='green', label='Lower Bound')
        plt.fill_between(self.dataset[:, 0], lower_bound, upper_bound, color='blue', alpha=0.1)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
