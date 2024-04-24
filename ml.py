import random
from typing import TypeVar, List, Tuple

X = TypeVar("X")
Y = TypeVar("Y")


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions of [prob, 1-prob]"""
    data = data.copy()  # make a copy because shuffle modifies the list
    random.shuffle(data)
    cut = int(len(data) * prob)
    return (data[:cut], data[cut:])


def train_test_split(
    xs: List[X], ys: List[Y], test_pct: float
) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    """Split xs and ys into (X_train, X_test, Y_train, Y_test)."""
    
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, prob=(1 - test_pct))
    X_train = [xs[i] for i in train_idxs]
    X_test = [xs[i] for i in test_idxs]
    Y_train = [ys[i] for i in train_idxs]
    Y_test = [ys[i] for i in test_idxs]
    return (X_train, X_test, Y_train, Y_test)


# test split_data
data = [n for n in range(1000)]
train, test = split_data(data, 0.75)
assert len(train) == 750
assert len(test) == 250
assert sorted(data) == sorted(train + test)


# test train_test_split
x = [n for n in range(1000)]
y = [n for n in range(1000)]
x_train, x_test, y_train, y_test = train_test_split(x, y, 0.30)
assert len(x_train) == len(y_train) == 700
assert len(x_test) == len(y_test) == 300
assert sorted(x) == sorted(x_train + x_test)
assert sorted(y) == sorted(y_train + y_test)




