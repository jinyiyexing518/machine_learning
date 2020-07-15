import math


def calc_entropy(dataSet):
    num_entries = len(dataSet)
    label_counts = {}
    for featVec in dataSet:
        current_label = featVec[-1]



