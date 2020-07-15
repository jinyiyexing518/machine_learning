from K_NN.demo2.get_and_display_data import file2matrix
import numpy as np


def auto_norm(dataset):
    min_value = dataset.min(axis=0)
    max_value = dataset.max(axis=0)
    ranges = max_value - min_value
    normal_dataset = np.zeros((np.shape(dataset)))
    m = dataset.shape[0]
    normal_dataset = dataset - np.tile(min_value, (m, 1))
    normal_dataset = normal_dataset / np.tile(ranges, (m, 1))
    return normal_dataset, ranges, min_value


if __name__ == "__main__":
    matrix, labels = file2matrix("./datingTestSet2.txt")
    normal_dataset, ranges, min_value = auto_norm(matrix)
    print(matrix, '\n', normal_dataset)


