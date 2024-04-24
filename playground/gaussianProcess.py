import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

x = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(x * np.sin(x))

plt.plot(x, y, label=r"$f(x) = x \ sin(x)$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")
plt.show()

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
x_train, y_train = x[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(x_train, y_train)
print(gp.kernel_)

mean_prediction, std_prediction = gp.predict(x, return_std=True)

plt.plot(x, y, label=r"$f(x) = x \ sin(x)$", linestyle="dotted")
plt.scatter(x_train, y_train, color="red", label="Training points")
plt.plot(x, mean_prediction, label="Mean prediction")
plt.fill_between(np.squeeze(x), mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, alpha=0.3, label=r"95% Confidence interval")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian Process Regression on noise-free data")
plt.show()

noise_std = .75
y_train_noisy = y_train + rng.normal(loc=0, scale=noise_std, size=y_train.size)
gp_noisy = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=10)
gp_noisy.fit(x_train, y_train_noisy)
print(gp_noisy.kernel_)

mean_prediction_noisy, std_prediction_noisy = gp_noisy.predict(x, return_std=True)

plt.plot(x, y, label=r"$f(x) = x \ sin(x)$", linestyle="dotted")
plt.errorbar(x_train, y_train_noisy, noise_std, linestyle="None", color="tab:blue", marker=".", markersize=10, label="Observations",)
plt.plot(x, mean_prediction_noisy, label="Mean prediction")
plt.fill_between(np.squeeze(x), mean_prediction_noisy - 1.96 * std_prediction_noisy, mean_prediction_noisy + 1.96 * std_prediction_noisy, alpha=0.3, label=r"95% Confidence interval")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian Process Regression on noisy data")
plt.show()

num_realizations = 10
realizations = gp.sample_y(x, num_realizations)
plt.figure(figsize=(10, 5))
for i in range(num_realizations):
    plt.plot(x, realizations[:, i], lw=1, linestyle='--')
plt.plot(x, y, label=r"$f(x) = x \ sin(x)$", linestyle="dotted")
plt.scatter(x_train, y_train, color="red", label="Observations")
plt.plot(x, mean_prediction, label="Mean prediction")
plt.fill_between(np.squeeze(x), mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, alpha=0.3, label=r"95% Confidence interval")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian Process Regression on noisy data with realizations")
plt.show()

