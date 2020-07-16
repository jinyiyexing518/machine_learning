import operator
from Decision_Tree.demo1.choose_best_feature_to_split import choose_best_feature_to_split
from Decision_Tree.demo1.split_dataSet import split_dataSet
from Decision_Tree.demo1.create_data import create_dataSet


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 1
        else:
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 数据集中的特征标签
    :return:
    """
    # class_list存放所有类标签
    class_list = [example[-1] for example in dataSet]
    # 第一个停止条件
    # 如果类标签列表中，计算序号为0的类标签数量
    # 如果和类标签列表长度相同
    # 则说明类别完全相同，无需再分
    # 返回该类别标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 第二个停止条件
    # 当遍历完所有特征，此时特征长度只剩一
    # 仍然不能将数据集划分为唯一类别的数组
    # 无法简单地返回唯一的类标签，返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majority_cnt(class_list)
    # choose_best_feature返回的是最佳特征的索引
    best_feature_index = choose_best_feature_to_split(dataSet)
    # 取得最佳特征的标签
    best_feature_label = labels[best_feature_index]
    myTree = {best_feature_label: {}}
    # 从labels中删除最佳特征
    del(labels[best_feature_index])
    feature_values = [example[best_feature_index] for example in dataSet]
    unique_values = set(feature_values)
    for value in unique_values:
        # 列表为引用类型数据，为防止改变原始数据，这里用sub_labels复制labels
        sub_labels = labels[:]
        # 递归调用，生成树
        # 每调用一次，labels中减少一个类标签，dataSet中的减少一列特征
        # 对于循环中的每一个value，根据选定特征是否等于value，split出子数据集
        myTree[best_feature_label][value] = create_tree(split_dataSet(dataSet, best_feature_index, value), sub_labels)
    return myTree


if __name__ == "__main__":
    dataSet, labels = create_dataSet()
    myTree = create_tree(dataSet, labels)
    print(myTree)

