import stats
import linealg
from typing import List, NamedTuple, Tuple, Dict
from linealg import Vector
from collections import Counter, defaultdict
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import random
import ml
import metrics


def majority_vote(labels: List[str]) -> str:
    """
    Return the mode of the labels.
    Assumes the labels are ordered from nearest to farthest.
    """
    counts = Counter(labels)
    winner, winner_count = counts.most_common(1)[0]
    num_winner = len([count for count in counts.values() if count == winner_count])
    if num_winner == 1:
        return winner
    # if there are 2 modes, try again without the farthest
    elif num_winner > 1:
        return majority_vote(labels[:-1])


class LabeledPoint(NamedTuple):
    point: Vector
    label: str


def knn_classify(k: int, labeled_points: LabeledPoint, new_point: Vector) -> str:
    # order labeled points from nearest to farthest
    by_distance = sorted(
        labeled_points, key=lambda lp: linealg.distance(lp.point, new_point)
    )
    # find the labels for k closest
    k_nearest = [lp.label for lp in by_distance[:k]]
    # and let them vote
    return majority_vote(k_nearest)


samples = [
    "a",
    "b",
    "c",
    "b",
    "a",
]
assert majority_vote(samples) == "b"

iris = load_iris()
xs = iris["data"]
ys = iris["target"]
iris_data = [LabeledPoint(x.tolist(), y) for (x, y) in zip(xs, ys)]
iris_train, iris_test = ml.split_data(iris_data, prob=0.7)

y_pred = [knn_classify(5, iris_train, test_point.point) for test_point in iris_test]
y_test = [lp.label for lp in iris_test]


# track how many times we see (predicted, actual)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0
for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label
    if predicted == actual:
        num_correct += 1
    confusion_matrix[(predicted, actual)] += 1
pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)
