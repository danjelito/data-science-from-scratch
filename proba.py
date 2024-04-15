import math
import matplotlib.pyplot as plt

SQRT_TWO_PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate the probability density function (PDF) of the normal distribution.
    PDF describes how the values of a random variable are distributed.
    E.g., PDF of a dice will be flat.

    Args:
    - x (float): The value at which to evaluate the PDF.
    - mu (float, optional): The mean of the normal distribution (default is 0).
    - sigma (float, optional): The standard deviation of the normal distribution (default is 1).

    Returns:
    float: The PDF value at x.

    Example:
    normal_pdf(0)  # With default mu=0 and sigma=1
    0.3989
    normal_pdf(2, mu=3, sigma=2)  # mu=3, sigma=2
    0.0267
    """
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    numerator = math.exp(exponent)  # e ^ exponent
    denominator = sigma * SQRT_TWO_PI
    return numerator / denominator


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate the cumulative distribution function (CDF) of the normal distribution.
    CDF gives the probability that the random variable takes on a value less than or equal to a given value.
    For instance, if you're looking at the CDF of a fair six-sided die roll,
    the value of the CDF at 3 would tell you the probability of rolling a number less than or equal to 3.

    Args:
    - x (float): The value at which to evaluate the CDF.
    - mu (float, optional): The mean of the normal distribution (default is 0).
    - sigma (float, optional): The standard deviation of the normal distribution (default is 1).

    Returns:
    float: The CDF value at x.

    Example:
    normal_cdf(0)  # With default mu=0 and sigma=1
    0.5000
    normal_cdf(2, mu=3, sigma=2)  # mu=3, sigma=2
    0.6915
    """
    numerator = x - mu
    denominator = sigma * math.sqrt(2)
    return 0.5 * (1 + math.erf(numerator / denominator))


def inverse_normal_cdf(
    p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001
) -> float:
    low_z = -10.0  # normal_cdf(-10) is very close to 0
    high_z = 10.0  # normal_cdf(10) is very close to 1
    while high_z - low_z > tolerance:
        mid_z = (low_z + high_z) / 2  # consider midpoint value
        mid_p = normal_cdf(mid_z, mu=0, sigma=1)  # get the probability of that midpoint
        if p < mid_p:
            high_z = mid_z  # desired p < resulted p, search below the mid_z
        elif p >= mid_p:
            low_z = mid_z  # desired p > resulted p, search above mid_z

    # if not standard, rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * mid_z
    return mid_z


def plot_pdf_example():
    xs = [x / 10 for x in range(-50, 50)]
    plt.plot(
        xs,
        [normal_pdf(x, mu=0, sigma=1) for x in xs],
        "red",
        label="mu 0 sigma 1 (standard normal)",
    )
    plt.plot(
        xs, [normal_pdf(x, mu=0, sigma=2) for x in xs], "green", label="mu 0 sigma 2"
    )
    plt.plot(
        xs, [normal_pdf(x, mu=0, sigma=0.5) for x in xs], "blue", label="mu 0 sigma 0.5"
    )
    plt.plot(
        xs, [normal_pdf(x, mu=-1, sigma=1) for x in xs], "black", label="mu -1 sigma 1"
    )
    plt.legend()
    plt.xlabel("value")
    plt.ylabel("probability")
    plt.title("various normal pdfs")
    plt.show()


def plot_cdf_example():
    xs = [x / 10 for x in range(-50, 50)]
    plt.plot(
        xs,
        [normal_cdf(x, mu=0, sigma=1) for x in xs],
        "red",
        label="mu 0 sigma 1 (standard normal)",
    )
    plt.plot(
        xs, [normal_cdf(x, mu=0, sigma=2) for x in xs], "green", label="mu 0 sigma 2"
    )
    plt.plot(
        xs, [normal_cdf(x, mu=0, sigma=0.5) for x in xs], "blue", label="mu 0 sigma 0.5"
    )
    plt.plot(
        xs, [normal_cdf(x, mu=-1, sigma=1) for x in xs], "black", label="mu -1 sigma 1"
    )
    plt.legend()
    plt.title("various normal cdfs")
    plt.xlabel("value")
    plt.ylabel("probability of getting value <=")
    plt.show()


assert 0.3988 < normal_pdf(0) < 0.3990
assert 0.84 < normal_cdf(1) < 0.85
assert 0.99 < inverse_normal_cdf(normal_cdf(1)) < 1.01
assert 492 < inverse_normal_cdf(p=0.3, mu=500, sigma=14.8) < 493

if __name__ == "__main__":
    # plot_pdf_example()
    # plot_cdf_example()
    print(normal_pdf(0))  # returns 0.3989
    print(normal_cdf(1))  # returns 0.8413447460685429
    print(inverse_normal_cdf(0.8413447460685429))  # should return 1
