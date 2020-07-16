from Decision_Tree.demo1.create_data import create_dataSet
import math


def calc_entropy(dataSet):
    num_entries = len(dataSet)
    label_counts = {}
    for feature_vector in dataSet:
        current_label = feature_vector[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 1
        else:
            label_counts[current_label] += 1
    entropy = 0.0
    for key in label_counts:
        # 计算出现概率
        probability = float(label_counts[key]) / num_entries
        # 计算信息熵
        # 增加种类，信息熵会变大
        entropy -= probability * math.log(probability, 2)
    return entropy


if __name__ == "__main__":
    myData, labels = create_dataSet()
    entropy = calc_entropy(myData)
    print("Entropy is {}".format(entropy))
