"""

"""
from math import log2


def calculate_entropy(labels):
    label_values = set(labels)
    label_counts = {}
    for item in label_values:
        label_counts[item] = 0
    for item in labels:
        label_counts[item] += 1

    entropy = 0

    for item in label_values:
        pi = label_counts[item]/len(labels)
        entropy += (-(pi * log2(pi)))
    return entropy


def calculate_gini(labels):
    label_values = set(labels)
    label_counts = {}
    for item in label_values:
        label_counts[item] = 0

    for item in labels:
        label_counts[item] += 1

    gini_sum = 0

    for item in label_values:
        pi = label_counts[item] / len(labels)
        gini_sum += pi * pi
    return 1 - gini_sum

def calculate_classification_error(labels):
    label_values = set(labels)
    label_counts = {}
    for item in label_values:
        label_counts[item] = 0

    for item in labels:
        label_counts[item] += 1

    pi_values = []

    for item in label_values:
        pi_values.append(label_counts[item] / len(labels))
    return 1 - max(pi_values)