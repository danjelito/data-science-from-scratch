tp = 70
fp = 4930
fn = 13930
tn = 981070


def accuracy(tp: float, fp: float, fn: float, tn: float) -> float:
    """ "Correct prediction / total prediction."""
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp: float, fp: float, fn: float, tn: float) -> float:
    """Percentage of **positive** predictions that are correct."""
    return tp / (tp + fp)


def recall(tp: float, fp: float, fn: float, tn: float) -> float:
    """Percentage of **positive** values correctly classified."""
    return tp / (tp + fn)


def f1(tp: float, fp: float, fn: float, tn: float) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)


assert accuracy(tp, fp, fn, tn) == 0.98114
assert precision(tp, fp, fn, tn) == 0.014
assert recall(tp, fp, fn, tn) == 0.005
