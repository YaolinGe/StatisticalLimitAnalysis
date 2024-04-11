from scipy.stats import norm


def std_dev_to_prob(std_dev):
    """
    Convert number of standard deviations (std_dev) to the corresponding
    probability in a confidence interval for a normal distribution.

    Parameters:
    std_dev (float): The number of standard deviations from the mean.

    Returns:
    float: The probability corresponding to the confidence interval
           defined by the number of standard deviations.
    """
    # Calculate the area to the left of the positive z-score
    area_left = norm.cdf(std_dev)

    # Calculate the probability of being within ±std_dev standard deviations
    # This is the area between the negative and positive z-scores
    probability = (area_left - (1 - area_left)) * 100  # Convert to percentage

    return probability


# Example usage:
std_dev = 2  # For a 95% confidence interval
probability = std_dev_to_prob(std_dev)
# print(f"The probability corresponding to ±{std_dev} standard deviations is approximately {probability:.2f}%.")

