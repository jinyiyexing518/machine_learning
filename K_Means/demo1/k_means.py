import numpy as np


def loadDataSet(fileName):
    """
    导入数据，此时无标签
    :param fileName:
    :return:
    """
    # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map all elements to float()
        fltLine = [float(item) for item in curLine]
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    """
    计算两个向量的欧氏距离
    :param vecA:
    :param vecB:
    :return:
    """
    return np.sqrt(sum(np.power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):
    """
    为给定数据集构建一个包含 k 个随机质心的集合
    找到数据集每一维特征的最小值和最大值，确保随点在数据的边界之内
    :param dataSet:
    :param k:
    :return:
    """
    n = np.shape(dataSet)[1]
    dataSet = np.array(dataSet)
    # 质心点数据
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    # create mat to assign data points
    # 两列：第一列记录簇索引值；第二列存储误差，即当前点到质心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    # 迭代标志变量 clusterChanged
    clusterChanged = True

    dataSet = np.array(dataSet)
    centroids = np.array(centroids)

    while clusterChanged:
        clusterChanged = False
        # 遍历每一个点，寻找最近质心
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            # 遍历质心，计算数据点到质心的距离
            for j in range(k):
                # 第 i 个数据点和第 j 个质心计算欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 距离比之前的小，则更换最小欧氏距离，并更换质心
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果质心改变，则迭代需要打开，置True
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 存储本次结果，第一列质心的索引，最短距离平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 打印此时的质心
        print(centroids)
        # 下面根据簇中的数据点的均值，更新质心位置
        for cent in range(k):
            # # 取出所有数据点的质心点，即第一列
            # a = clusterAssment[:, 0]
            # # 取出属于当前质心的数据，置为True
            # b = a.A == cent
            # # 获得索引值，由于是二维的矩阵，所以这里的索引分为行索引和列索引
            # c = np.nonzero(b)
            # # 由于只需要行索引即可，这里取出行索引，列索引均为第 0 列
            # d = c[0]
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算均值，将计算的均值作为新的质心，如果迭代标志变量仍然为True，返回上面继续迭代
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


if __name__ == "__main__":
    dataMat = loadDataSet('./testSet.txt')
    # centroids = randCent(dataMat, 2)
    # print(centroids)

    centroids, clusterAssment = kMeans(dataMat, 2, distMeas=distEclud, createCent=randCent)

