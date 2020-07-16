from Decision_Tree.demo1.calculate_data_entropy import calc_entropy
from Decision_Tree.demo1.split_dataSet import split_dataSet


def choose_best_feature_to_split(dataSet):
    feature_num = len(dataSet[0]) - 1
    base_entropy = calc_entropy(dataSet)
    best_inf_gain = 0.0
    best_feature = -1
    # 遍历特征
    for i in range(feature_num):
        # dataSet中某一特征下，所有数据的该特征以列表形式保存再feature_list中
        feature_list = [example[i] for example in dataSet]
        # 以集合形式，保存不一样的特征值
        unique_values = set(feature_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_dataSet = split_dataSet(dataSet, i, value)
            probability = len(sub_dataSet) / float(len(dataSet))
            new_entropy += probability * calc_entropy(sub_dataSet)
        # 以无序变有序，划分的子数据集中的种类更少
        # 所以熵减小，意味着获取了信息量，不确定性减小
        info_gain = base_entropy - new_entropy
        if (info_gain > best_inf_gain):
            best_inf_gain = info_gain
            best_feature = i
    # 返回按照最佳特征，以此特征划分可以获得最大信息熵增益
    return best_feature

