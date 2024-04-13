from collections import Counter
from typing import List
import math

import linealg


def mean(xs: List[int | float]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    Example:
    mean([1, 2, 3]) => 2.0
    """
    return sum(xs) / len(xs)


def _median_even(xs: List[int | float]) -> float:
    """
    Calculate the median of a list with an even number of elements.

    Example:
    _median_even([1, 2, 3, 4]) => 2.5
    """
    midpoint = len(xs) // 2
    return (sorted(xs)[midpoint - 1] + sorted(xs)[midpoint]) / 2


def _median_odd(xs: List[int | float]) -> float:
    """
    Calculate the median of a list with an odd number of elements.

    Example:
    _median_odd([1, 2, 3]) => 2
    """
    return sorted(xs)[len(xs) // 2]


def median(xs: List[int | float]) -> float:
    """
    Calculate the median of a list of numbers.

    Example:
    median([1, 2, 3]) => 2.0
    """
    return _median_even(xs) if len(xs) % 2 == 0 else _median_odd(xs)


def quantile(xs: List[int | float], p: float) -> float:
    """
    Calculate the p-th percentile of a list of numbers.

    Example:
    quantile([1, 2, 3, 4], 0.5) => 3
    """
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def mode(xs: List[int | float]) -> List[float]:
    """
    Find the mode(s) of a list of numbers.

    Example:
    mode([1, 1, 2, 2, 3, 4]) => [1, 2]
    """
    counts = Counter(xs)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]


def data_range(xs: List[int | float]) -> float:
    """
    Calculate the range of a list of numbers.

    Example:
    data_range([1, 2, 10]) => 9
    """
    return max(xs) - min(xs)


def de_mean(xs: List[int | float]) -> List[float]:
    """
    Subtract mean from each element to make the resulting vector's mean = 0.

    Example:
    de_mean([1, 2, 3]) => [-1, 0, 1]
    """
    return [x_i - mean(xs) for x_i in xs]


def variance(xs: List[int | float]) -> List[float]:
    """
    Calculate the sample variance of a list of numbers.

    Example:
    variance([1, 2, 3]) => 1.0
    """
    n = len(xs)
    return linealg.sum_of_squares(de_mean(xs)) / (n - 1)


def standard_deviation(xs: List[int | float]) -> List[float]:
    """
    Calculate the sample standard deviation of a list of numbers.

    Example:
    standard_deviation([1, 2, 3]) => 1.0
    """
    return math.sqrt(variance(xs))


def interquartile_range(xs: List[int | float]) -> List[float]:
    """
    Calculate the interquartile range of a list of numbers.

    Example:
    interquartile_range([1, 2, 3, 4, 5, 6]) => 3
    """
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def covariance(xs: List[int | float], ys: List[int | float]):
    """
    Calculate the covariance between two lists of numbers.

    Example:
    covariance([1, 2, 3], [10, 20, 27]) => 8.5
    """
    assert len(xs) == len(ys), "xs and ys must be of the same length."
    n = len(xs)
    return linealg.dot(de_mean(xs), de_mean(ys)) / (n - 1)


def correlation(xs: List[int | float], ys: List[int | float]):
    """
    Calculate the correlation between two lists of numbers.

    Example:
    correlation([41, 19, 23, 40, 55, 57, 33], [94, 60, 74, 71, 82, 76, 61]) => ~0.536
    """
    return covariance(xs, ys) / (standard_deviation(xs) * standard_deviation(ys))


assert mean([1, 2, 3]) == 2
assert median([1, 2, 3]) == 2
assert median([1, 2, 3, 4]) == 2.5
assert quantile([1, 2, 3, 4], 0.5) == 3
assert mode([1, 1, 2, 2, 3, 4]) == [1, 2]
assert data_range([1, 2, 10]) == 9
assert de_mean([1, 2, 3]) == [-1, 0, 1]
assert variance([1, 2, 3]) == 1
assert standard_deviation([1, 2, 3]) == 1
assert covariance([1, 2, 3], [10, 20, 27]) == 8.5
assert (
    0.53
    < (correlation([41, 19, 23, 40, 55, 57, 33], [94, 60, 74, 71, 82, 76, 61]))
    < 0.54
)