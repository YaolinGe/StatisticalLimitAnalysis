import numpy as np

def replicate_signals(signal, number_of_replicas=1, noise_level=0.1, noise_seed=42, time_shift_range:int=5):
    np.random.seed(noise_seed)
    replicated_signals = []
    for _ in range(number_of_replicas):
        time_shift = np.random.choice([-time_shift_range, time_shift_range])
        shifted_signal = np.roll(signal, time_shift, axis=0)
        noise = np.random.normal(0, noise_level, signal.shape[0]) * np.amax(signal[:, 1])
        replicate = shifted_signal[:, 1] + noise
        replicated_signals.append(replicate)

    return np.array(replicated_signals)

