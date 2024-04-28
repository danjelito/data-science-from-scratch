from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import math
from collections import defaultdict
import re


def tokenize(text: str) -> Set[str]:
    text = text.lower()  # lowercase
    all_words = re.findall("[a-z0-9']+", text)  # split
    return set(all_words)  # remove duplicates


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayes:

    def __init__(self, k: float) -> None:
        self.k = k  # smoothing factor to avoid probability = 0
        self.tokens: Set[str] = set()  # empty set
        self.token_ham_counts: Dict[str, float] = defaultdict(int)
        self.token_spam_counts: Dict[str, float] = defaultdict(int)
        self.count_ham_messages = 0
        self.count_spam_messages = 0

    def train(self, messages=Iterable[Message]) -> None:
        for message in messages:
            # count spam and ham
            if message.is_spam:
                self.count_spam_messages += 1
            elif not message.is_spam:
                self.count_ham_messages += 1
            # count each word of spam and ham
            tokens = tokenize(message.text)
            for token in tokens:
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                elif not message.is_spam:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Returns P(token|spam) and P(token|ham)"""
        count_this_word_in_spam = self.token_spam_counts.get(token, 0)
        count_this_word_in_ham = self.token_ham_counts.get(token, 0)
        p_token_spam = (count_this_word_in_spam + self.k) / (
            self.count_spam_messages + (2 * self.k)
        )
        p_token_ham = (count_this_word_in_ham + self.k) / (
            self.count_ham_messages + (2 * self.k)
        )
        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = 0
        log_prob_if_ham = 0

        # iterate through each word in vocab
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            # if this token in message
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            # if this token not in message
            # add the log probability of not seeing it
            elif not token in text_tokens:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


messages = [
    Message("spam rules", is_spam=True),
    Message("ham rules", is_spam=False),
    Message("hello ham", is_spam=False),
]
model = NaiveBayes(0.5)
model.train(messages)
assert model.tokens == {"spam", "rules", "ham", "hello"}
assert tokenize("Data Science is science") == {"data", "science", "is"}
assert model.count_spam_messages == 1
assert model.count_ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"
k = 0.5
probs_if_spam = [
    (1 + k) / (1 + 2 * k),  # "spam" present
    1 - ((0 + k) / (1 + 2 * k)),  # "ham" not present
    1 - ((1 + k) / (1 + 2 * k)),  # "rules" not present
    (0 + k) / (1 + 2 * k),  # "hello" present
]
probs_if_ham = [
    (0 + k) / (2 + 2 * k),  # "spam" present
    1 - ((2 + k) / (2 + 2 * k)),  # "ham" not present
    1 - ((1 + k) / (2 + 2 * k)),  # "rules" not present
    (1 + k) / (2 + 2 * k),  # "hello" present
]
p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)

print(p_if_spam / (p_if_spam + p_if_ham))
print(model.predict(text))
