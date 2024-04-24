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
        signal_random = self.signal.generate_random_signal()
        plt.figure()
        plt.plot(self.signal.x, signal_random, label='Random Signal')
        plt.scatter(self.signal.x_train, self.signal.y_train, color="red", label="Training points")
        plt.plot(self.signal.x, self.signal.y, label="True Signal")
        plt.legend()
        plt.show()


