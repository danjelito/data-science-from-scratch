from typing import Tuple
import proba as proba
import math
import random
import numpy as np

random.seed(0)
np.random.seed(0)

# say we want to test if a coin is fair
# h0 : the coin is fair
# ha : the coin is not fair


def normal_approximation_to_binomial(n: int, p: float):
    """
    Approximate the binomial distribution with a normal distribution.
    The function calculates the mean (mu) and standard deviation (sigma) of the
    normal distribution that approximates the binomial distribution with parameters n and p.
    The mean is given by mu = n * p and the standard deviation by sigma = sqrt(n * p * (1 - p)).

    Parameters:
    n (int): Number of trials.
    p (float): Probability of success on each trial.

    Returns:
    Tuple[float, float]: Mean (mu) and standard deviation (sigma) of the normal approximation.
    """
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


# proba that var is below threshold
normal_proba_below = proba.normal_cdf


def normal_proba_above(x: float, mu: float = 0, sigma: float = 1):
    """
    Calculate the probability that a variable is above a given threshold.

    Args:
        x (float): The threshold value.
        mu (float, optional): The mean of the normal distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the normal distribution. Defaults to 1.

    Returns:
        float: The probability that the variable is above the threshold.
    """
    return 1 - proba.normal_cdf(x, mu, sigma)


def normal_proba_between(lo: float, hi: float, mu: float = 0, sigma: float = 1):
    """
    Calculate the probability that a variable is between two thresholds.

    Args:
        lo (float): The lower threshold value.
        hi (float): The upper threshold value.
        mu (float, optional): The mean of the normal distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the normal distribution. Defaults to 1.

    Returns:
        float: The probability that the variable is between the two thresholds.
    """
    return normal_proba_above(hi, mu, sigma) - normal_proba_above(lo, mu, sigma)


def normal_proba_outside(lo: float, hi: float, mu: float = 0, sigma: float = 1):
    """
    Calculate the probability that a variable is outside two thresholds.

    Args:
        lo (float): The lower threshold value.
        hi (float): The upper threshold value.
        mu (float, optional): The mean of the normal distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the normal distribution. Defaults to 1.

    Returns:
        float: The probability that the variable is outside the two thresholds.
    """
    return 1 - normal_proba_between(lo, hi, mu, sigma)


def normal_upper_bound(probability, mu, sigma) -> float:
    """
    Returns the z for which P(Z <= z) given a probability.

    Args:
        probability (float): The probability value.
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.

    Returns:
        float: The upper bound z-value.
    """
    return proba.inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability, mu, sigma) -> float:
    """
    Returns the z for which P(Z >= z) given a probability.

    Args:
        probability (float): The probability value.
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.

    Returns:
        float: The lower bound z-value.
    """
    return proba.inverse_normal_cdf((1 - probability), mu, sigma)


def normal_two_sided_bound(probability, mu, sigma) -> Tuple[float, float]:
    """
    Returns the lower and upper bounds for a two-sided test given a probability.

    Args:
        probability (float): The probability value.
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.

    Returns:
        Tuple[float, float]: The lower and upper bound z-values.
    """
    tail_probability = probability / 2
    # upper bound should have proba above it
    upper = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have proba below it
    lower = normal_upper_bound(tail_probability, mu, sigma)
    return lower, upper


def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate the two-sided p-value given a test statistic.

    Args:
        x (float): The test statistic.
        mu (float, optional): The mean of the normal distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the normal distribution. Defaults to 1.

    Returns:
        float: The two-sided p-value.
    """
    if x < mu:  # x is lower than mean
        return 2 * normal_proba_below(x, mu, sigma)
    else:  # x is higher than mean
        return 2 * normal_proba_above(x, mu, sigma)


if __name__ == "__main__":

    def by_stats():
        # say we do the coin flip 1000x
        mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

        # with 5% alpha, the lower and upper bound of the experiment is given by
        alpha = 0.05
        lower, upper = normal_two_sided_bound(alpha, mu_0, sigma_0)

        # say we observe 530 head. the p-value is
        x = 529.5
        p_value = two_sided_p_value(x, mu_0, sigma_0)

        print("======================== With statistics ========================")
        print(
            f"If coin is fair, X should have (around) mean {mu_0} and SD {sigma_0: .1f}"
        )
        print(f"To reject h0, the bounds are {lower: .1f} and {upper: .1f}")
        print(
            "Assuming h0 is true, there is 5% chance we observe X outside this bound."
        )
        print(f"With 530 head, the p-value is {p_value}")
        # say we observe 530 head. the p-value is
        x = 529.5

    def by_permutation():
        print("======================== With permutation ========================")
        # say we do the coin flip 1000x
        mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
        print(
            f"If coin is fair, X should have (around) mean {mu_0} and SD {sigma_0: .1f}"
        )

        n_experiment = 10_000
        n_coin_taken = 1000
        n_heads = []
        for _ in range(n_experiment):  # 10_000 experiment
            n_head = sum(
                np.random.choice([0, 1], size=(n_coin_taken), replace=True)
            )  # number of head taken from 1000x
            n_heads.append(n_head)

        # say we observe 530 head. the p-value is
        x = 529.5
        p_value = np.mean([n > x for n in n_heads]) * 2

        alpha = 0.05
        lower = np.percentile(n_heads, (alpha * 100 / 2))
        upper = np.percentile(n_heads, (100 - (alpha * 100 / 2)))
        print(f"To reject h0, the bounds are {lower} and {upper}")
        print(
            "Assuming h0 is true, there is 5% chance we observe X outside this bound."
        )
        print(f"With 530 head, the p-value is {p_value}")

    print("\n")
    by_stats()
    print("\n")
    by_permutation()
    print("\n")
