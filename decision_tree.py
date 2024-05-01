from typing import List, Any, NamedTuple, Optional, TypeVar, Dict, DefaultDict
from collections import Counter
import math


def entropy(class_proba: List[float]) -> float:
    """Given a list of proba, compute the entropy.
    Ignore proba = 0.
    """
    return sum(-p * math.log(p, 2) for p in class_proba if p > 0)


def class_proba(labels: List[Any]) -> List[float]:
    """Given a list of labels, returns proportion of each class
    to the whole classes.
    """
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
    """Calculates entropy from a list of labels."""
    return entropy(class_proba(labels))


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Weighted sum of data_entropy.
    Think of it as a node that splits the data into 2 partitions
    then we want to find the entropy of these 2 partitions.
    """
    total_count = sum(len(subsets) for subsets in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)


# imagine a node that separates the labels into 2 classes with probas as below
assert entropy([1.0, 0.0]) == 0  # low entropy, which is good
assert 0.81 < entropy([0.25, 0.75]) < 0.82  # medium entropy, which is bad
assert entropy([0.5, 0.5]) == 1  # max entropy, which is very bad

# imagine if a node outputs 4 samples as
samples = ["class_a", "class_b", "class_b", "class_b"]
assert class_proba(samples) == [0.25, 0.75]
assert data_entropy(samples) == entropy([0.25, 0.75])
assert 0.81 < data_entropy(samples) < 0.82  # medium entropy, which is bad


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # this is our target


T = TypeVar("T")

inputs = [
    Candidate("Senior", "Java", False, False, False),
    Candidate("Senior", "Java", False, True, False),
    Candidate("Mid", "Python", False, False, True),
    Candidate("Junior", "Python", False, False, True),
    Candidate("Junior", "R", True, False, True),
    Candidate("Junior", "R", True, True, False),
    Candidate("Mid", "R", True, True, True),
    Candidate("Senior", "Python", False, False, False),
    Candidate("Senior", "R", True, False, True),
    Candidate("Junior", "Python", True, False, True),
    Candidate("Senior", "Python", True, True, True),
    Candidate("Mid", "Python", False, True, True),
    Candidate("Mid", "Java", True, False, True),
    Candidate("Junior", "Python", False, True, False),
]


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on specific attribute."""
    partitions: Dict[Any, List[T]] = DefaultDict(list)
    for input in inputs:
        key = getattr(input, attribute)  # value of the specific attribute
        partitions[key].append(input)
    return partitions


def partition_entropy_by(
    inputs: List[Any], attribute: str, label_attribute: str
) -> float:
    """Compute the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    # get only the values of label_attribute
    labels = [
        [getattr(input, label_attribute) for input in partition]
        for partition in partitions.values()
    ]
    return partition_entropy(labels)


assert 0.69 < partition_entropy_by(inputs, "level", "did_well") < 0.70
assert 0.86 < partition_entropy_by(inputs, "lang", "did_well") < 0.87
assert 0.78 < partition_entropy_by(inputs, "tweets", "did_well") < 0.79
assert 0.89 < partition_entropy_by(inputs, "phd", "did_well") < 0.90

# see samples
sample = partition_by(inputs, "level")
print(sample)
sample = [
    [getattr(input, "did_well") for input in partition] for partition in sample.values()
]
print(sample)
sample = partition_entropy(sample)
print(sample)


for key in ["level", "lang", "tweets", "phd"]:
    print(key, partition_entropy_by(inputs, key, "did_well"))

senior_inputs = [input for input in inputs if input.level == "Senior"]
assert 0.4 == partition_entropy_by(senior_inputs, "lang", "did_well")
assert 0.0 == partition_entropy_by(senior_inputs, "tweets", "did_well")
assert 0.95 < partition_entropy_by(senior_inputs, "phd", "did_well") < 0.96


from typing import NamedTuple, Union, Any


class Leaf(NamedTuple):
    value: Any


class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[Leaf, Split]

hiring_tree = Split(
    # attribute
    "level",
    # subtrees
    {
        "Junior": Split("phd", {False: Leaf(True), True: Leaf(False)}),
        "Mid": Leaf(True),
        "Senior": Split("tweets", {False: Leaf(False), True: Leaf(True)}),
    },
)


def classify(tree: DecisionTree, input: Any):
    """Classify the input using the given decision tree."""

    # if leaf node, returns its value
    if isinstance(tree, Leaf):
        return tree.value

    # otherwise this tree consists of an attriute to split on
    subtree_key = getattr(input, tree.attribute)

    # if no subtree (i.e. this value is never seen on training data)
    if subtree_key not in tree.subtrees:
        return tree.default_value

    subtree = tree.subtrees[subtree_key]
    return classify(subtree, input)


def build_tree_id3(
    inputs: List[Any], split_attributes: List[str], target_attribute: str
) -> DecisionTree:
    # count target labels
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]
    # if there is unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)
    # if no split attributes left, return majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # otherwise, split by the best attribute
    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute."""
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # recursively build the subtrees
    subtrees = {
        attribute_value: build_tree_id3(subset, new_attributes, target_attribute)
        for attribute_value, subset in partitions.items()
    }
    return Split(best_attribute, subtrees, default_value=most_common_label)


tree = build_tree_id3(inputs, ["level", "lang", "tweets", "phd"], "did_well")
# Should predict True
assert classify(tree, Candidate("Junior", "Java", True, False))
# Should predict False
assert not classify(tree, Candidate("Junior", "Java", True, True))
# Should predict True
assert classify(tree, Candidate("Intern", "Java", True, True))
