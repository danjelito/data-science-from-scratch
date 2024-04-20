from typing import Tuple, List
import linealg
import stats
from linealg import Vector
from sklearn.preprocessing import StandardScaler


def calculate_mean_std(data: List[Vector]) -> Tuple[List, List]:
    num_element = len(data[0])
    means = linealg.vector_mean(data)
    stds = [
        stats.standard_deviation([v[i] for v in data])
        for i in range(num_element)
    ]
    return means, stds


def rescale(data: List[Vector]) -> List[Vector]:
    means, stds = calculate_mean_std(data)
    num_element = len(data[0])
    rescaled_data = data.copy()
    for v in rescaled_data:
        for i in range(num_element):
            if stds[i] > 0:
                v[i] = (v[i] - means[i]) / stds[i]

    return rescaled_data



data = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
print(rescale(data))
means, stds = calculate_mean_std(rescale(data))
assert means == [0, 0, 1]
assert stds == [1, 1, 0]


















