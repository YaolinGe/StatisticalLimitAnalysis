"""
This test file will test the functions in the Signal.py file.
"""
import matplotlib.pyplot as plt
from unittest import TestCase
from Signal.Signal import Signal


class TestSignal(TestCase):

    def setUp(self):
        self.signal = Signal()

    def test_generate_random_signal(self):
        # signal_random = self.signal.generate_random_signal()
        signal_random = self.signal.generate_signal_using_gaussian_process()
        plt.figure()
        plt.plot(self.signal.x, signal_random, label='Random Signal')
        plt.scatter(self.signal.x_train, self.signal.y_train, color="red", label="Training points")
        plt.plot(self.signal.x, self.signal.y, label="True Signal")
        confidence_interval = self.signal.get_confidence_interval(0.95)

        plt.fill_between(self.signal.x.flatten(), confidence_interval[:, 0], confidence_interval[:, 1], alpha=0.3,
                         label="95% Confidence Interval")
        plt.legend()
        plt.show()

    # def test_generate_signal_using_arima(self):
    #     signal_arima = self.signal.generate_signal_using_arima()
    #     plt.figure()
    #     plt.plot(self.signal.x, signal_arima, label='ARIMA Signal')
    #     plt.scatter(self.signal.x_train, self.signal.y_train, color="red", label="Training points")
    #     plt.plot(self.signal.x, self.signal.y, label="True Signal")
    #     plt.legend()
    #     plt.show()
    #
    # def test_generate_signal_using_hmm(self):
    #     signal_hmm = self.signal.generate_signal_using_hmm()
    #     plt.figure()
    #     plt.plot(self.signal.x, signal_hmm, label='HMM Signal')
    #     plt.scatter(self.signal.x_train, self.signal.y_train, color="red", label="Training points")
    #     plt.plot(self.signal.x, self.signal.y, label="True Signal")
    #     plt.legend()
    #     plt.show()
    #
    # def test_generate_signal_using_kalman_filter(self):
    #     signal_kalman_filter = self.signal.generate_signal_using_kalman_filter()
    #     plt.figure()
    #     plt.plot(self.signal.x, signal_kalman_filter, label='Kalman Filter Signal')
    #     plt.scatter(self.signal.x_train, self.signal.y_train, color="red", label="Training points")
    #     plt.plot(self.signal.x, self.signal.y, label="True Signal")
    #     plt.legend()
    #     plt.show()

