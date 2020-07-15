import math


def calc_entropy(dataSet):
    num_entries = len(dataSet)
    label_counts = {}
    for featVec in dataSet:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        else:
            label_counts[current_label] += 1



