import numpy as np
import matplotlib.pyplot as plt


def load_dataSet():
    """
    读取数据文件
    :return: 返回两个数据列表
    """
    data_matrix = []
    label_matrix = []
    fr = open('./testSet.txt')
    for line in fr.readlines():
        # strip和split都有默认，strip默认为空格，换行，split默认空格，换行，table
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix


def sigmoid(intX):
    """
    sigmoid函数
    :param intX: 输入
    :return: sigmoid输出
    """
    # 因为这里inX是numpy的矩阵，所以需要numpy.exp
    return 1.0 / (1 + np.exp(-intX))


def gradient_ascent(data_matrix_in, class_labels):
    """
    梯度上升函数
    :param data_matrix_in:输入的数据
    :param class_labels: 输入的标签
    :return: 经过循环后的最优参数
    """
    # 转换成矩阵形式
    data_matrix = np.mat(data_matrix_in)
    # transpose矩阵转置
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    # 更新的步长（学习率）
    alpha = 0.001
    max_cycle = 500
    weights = np.ones((n, 1))
    for k in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def random_gradient_ascent(data_matrix, class_labels):
    """
    随机梯度上升算法,一次用一个样本来更新回归系数
    :param data_matrix:
    :param class_labels:
    :return:
    """
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def random_gradient_ascent_modify(data_matrix, class_labels, num_iter=150):
    """
    改进的梯度上升算法，防止由于不能正确分类的样本点引起的振荡
    :param data_matrix:
    :param class_labels:
    :return:
    """
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(np.arange(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(np.random.uniform(0, len(data_index)))
            index = data_index[rand_index]
            del (data_index[rand_index])
            h = sigmoid(sum(data_matrix[index] * weights))
            error = class_labels[index] - h
            weights = weights + alpha * error * data_matrix[index]
    return weights


def plot_best_fit(wei):
    """
    绘制最佳拟合直线
    :param wei: 系数矩阵
    :return: None
    """
    # getA将矩阵转化为数组
    # weights = wei.getA()
    weights = np.array(wei)
    data_matrix, label_matrix = load_dataSet()
    data_array = np.array(data_matrix)
    n = np.shape(data_array)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_matrix[i]) == 1:
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    # 获取数据
    data_matrix, label_matrix = load_dataSet()

    # 梯度上升算法
    weights1 = gradient_ascent(data_matrix, label_matrix)
    print(weights1)
    # 绘制最佳拟合直线
    plot_best_fit(weights1)

    # 随机梯度上升算法，缺点：单个数据引导梯度上升，不能正确分类的样本点会引起振荡
    weights2 = random_gradient_ascent(np.array(data_matrix), label_matrix)
    print(weights2)
    # 绘制最佳拟合直线
    plot_best_fit(weights2)

    # 改进的随机梯度上升算法
    weights3 = random_gradient_ascent_modify(np.array(data_matrix), label_matrix)
    print(weights3)
    # 绘制最佳拟合直线
    plot_best_fit(weights3)








