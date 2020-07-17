import numpy as np
import math


def load_dataSet():
    """
    加载数据
    :return: 返回数据以及标签
    """
    # 切分的词条
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vector


def create_vocabulary_list(dataSet):
    """
    创建词汇表
    :param dataSet: 输入的所有数据
    :return: 词汇表
    """
    vocabulary_Set = set([])
    for document in dataSet:
        vocabulary_Set = vocabulary_Set | set(document)
    return list(vocabulary_Set)


def set_words_to_vector(vocabulary_list, inputSet):
    """
    将原始数据转换成向量表，其中词汇表中每一个位置的词汇是否出现，用1和0代替
    :param vocabulary_list: 词汇表
    :param inputSet: 输入的数据（一行）
    :return:
    """
    return_vector = [0] * len(vocabulary_list)
    for word in inputSet:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] = 1
        else:
            print("The word {} is not in my vocabulary!".format(word))
    return return_vector


def train_naive_bayes(train_matrix, train_category):
    """
    :param train_matrix: 输入文档，一行行字符列表
    :param train_category: 输入文档类别标签
    :return:
    """
    num_train_datas = len(train_matrix)
    num_words = len(train_matrix[0])
    # 计算class=1，即侮辱性文档的概率
    pAbusive = sum(train_category) / float(len(train_matrix))
    # 这里为了防止零值出现
    p0Num = np.ones(num_words)
    p1Num = np.ones(num_words)
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历所有数据
    for i in range(num_train_datas):
        # 计算p(w0|1),p(w1|1)...p(wN|1)
        if train_category[i] == 1:
            # 计算侮辱性类别中每个word的频数，以列表形式
            p1Num += train_matrix[i]
            # 计算侮辱性类别数据总数
            # 这里按理说应该+1来计算条件概率，但是却是加上所有word数
            # 不过因为是作为分母乘积形式，对比较影响不大
            # p1Denom += 1
            p1Denom += sum(train_matrix[i])
        else:
            p0Num += train_matrix[i]
            # p0Denom += 1
            p0Denom += sum(train_matrix[i])
    p1Vector = np.log(p1Num / p1Denom)
    p0Vector = np.log(p0Num / p0Denom)
    return p0Vector, p1Vector, pAbusive


if __name__ == "__main__":
    posting_list, class_vector = load_dataSet()
    vocabulary_list = create_vocabulary_list(posting_list)
    print(vocabulary_list)
    return_vector = set_words_to_vector(vocabulary_list, posting_list[0])
    print(return_vector)

    train_mat = []
    for post in posting_list:
        train_mat.append(set_words_to_vector(vocabulary_list, post))
    p0Vector, p1Vector, pAbusive = train_naive_bayes(train_mat, class_vector)

    print('\n', '\n', '\n', p0Vector, '\n', '\n', '\n', p1Vector, '\n', '\n', '\n', pAbusive)

