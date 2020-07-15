import numpy as np
import operator


def createDataSet():
    """
    function： 原始数据集
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    function: k近邻算法实现
    :param inX: 用于分类的输入向量(数组)
    :param dataSet: 输入的原始样本集（数组）
    :param labels: 原始样本集的标签（列表）
    :param k: 邻居数（整型数int）
    :return: 返回预测的类别
    """
    dataSetsize = dataSet.shape[0]
    # tile函数拼接dataSetsize行，1列个inX，目的是为了与dataSet等同维度，以做差比较
    diffMat = np.tile(inX, (dataSetsize, 1)) - dataSet
    # 计算距离,差平方，求和，开根号
    sqDiffMat = diffMat ** 2
    sqdistances = sqDiffMat.sum(axis=1)
    distances = sqdistances ** 0.5
    # argsort返回distances中距离从小到大的索引值
    sortedDistIndicies = distances.argsort()
    # 创建空字典
    classCount = {}
    for i in range(k):
        # 依次取得距离最近的前 k 个数据的标签
        votelabel = labels[sortedDistIndicies[i]]
        # 字典中计算前 k 个数据的不同的类别出现的次数
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回前 k 个数据中，出现次数最多的类别作为测试数据的类别
    return sortedClassCount[0][0]


if __name__ == "__main__":
    group, labels = createDataSet()
    inX = [0.1, 0.1]
    predict_classification = classify0(inX, group, labels, 3)
    print("预测类别为：{}".format(predict_classification))



