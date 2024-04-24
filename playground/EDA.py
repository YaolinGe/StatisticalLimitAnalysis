import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from tdqm import tdqm

#%%
# prompt a user input to get the file path
file_path = r"C:\Users\nq9093\Downloads\ExportedFiles_20240105_024938\CoroPlus_230912-145816.cut"

# parse the file using the cut file parser located in .\tools\CutFileParser.exe
os.system(r".\tools\CutFileParserCLI.exe " + file_path)

#%%
# Load csv file ignoreing the first row, and sep is a comma

# def load_data(filename):
#     return pd.read_csv(r"C:\Users\nq9093\AppData\Local\Temp\CutFileParser\"+filename, skiprows=1, sep=",", header=None)

load = pd.read_csv(r"C:\Users\nq9093\AppData\Local\Temp\CutFileParser\Load.csv", skiprows=1, sep=",", header=None).to_numpy()
temperature = pd.read_csv(r"C:\Users\nq9093\AppData\Local\Temp\CutFileParser\Temperature.csv", skiprows=1, sep=",", header=None).to_numpy()
deflection = pd.read_csv(r"C:\Users\nq9093\AppData\Local\Temp\CutFileParser\Deflection.csv", skiprows=1, sep=",", header=None).to_numpy()
#%%
plt.subplot(311)
plt.plot(load[:,0], load[:,1])
plt.title("Load")

plt.subplot(312)
plt.plot(temperature[:,0], temperature[:,1])
plt.title("Temperature")

plt.subplot(313)
plt.plot(deflection[:,0], deflection[:,1])
plt.title("Deflection")

plt.tight_layout()
plt.show()

#%% populate 100 replicates based on the data


def calculate_mean_and_std(data):
    """
    This function will populate 100 replicates with added noise and return the mean and standard deviation of the replicates.

    Parameters:
    - data: A 2D numpy array where the second column contains the values to which noise is added.

    Returns:
    - mean: Mean of the replicates across the added noise.
    - std: Standard deviation of the replicates across the added noise.
    """
    replicates = []

    # Ensure data is a numpy array with the correct dtype for numerical stability
    data = np.asarray(data, dtype=np.float64)

    # Create 100 replicates with added noise
    for i in range(100):
        print(i)
        # Add random noise to the data
        shift = np.random.choice([-5, 5])
        shifted_data = np.roll(data, shift)
        noise = np.random.normal(0, .1, size=data.shape[0]) * np.max(data)
        replicate = shifted_data + noise
        replicates.append(replicate)

    replicates = np.array(replicates, dtype=np.float64)

    # Calculate the mean and standard deviation of the replicates
    mean = np.mean(replicates, axis=0)
    std = np.std(replicates, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation

    return mean, std


#%%

# calculate mean and std for temperature


data_raw = load
data = data_raw[:, 1]

# data_mean, data_std = calculate_mean_and_std(temperature[:, 1])
data_mean, data_std = calculate_mean_and_std(data)

plt.figure()
plt.plot(data)
plt.show()

#%% plot the error bar for temperature replicated data with noise added with confidence interval of 95%

import seaborn as sns
# plt.figure()
# plt.subplot(121)
# plt.plot(temperature[:, 0], temperature[:, 1])
# plt.title("Temperature")
#
# plt.subplot(122)

def plot_with_number_of_sigma(sigma):
    """ Plot the temperature with the number of sigma using random color """
    rnd = np.random.rand(3,)
    sns.lineplot(x=data_raw[:, 0], y=data_mean + sigma * data_std, color=rnd, alpha=.25)
    sns.lineplot(x=data_raw[:, 0], y=data_mean - sigma * data_std, color=rnd, alpha=.25)
    plt.fill_between(data_raw[:, 0], data_mean - sigma * data_std, data_mean + sigma * data_std, color=rnd, alpha=0.05)

plt.figure()
sns.lineplot(x=data_raw[:, 0], y=data_mean, color='red')
[plot_with_number_of_sigma(sigma) for sigma in range(1, 6)]
plt.xlabel("Time")
plt.show()


#%%
plt.figure()
plt.plot(data_std)
plt.title("Standard deviation of temperature")
plt.show()




