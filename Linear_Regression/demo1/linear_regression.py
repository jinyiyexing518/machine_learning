import numpy as np


def loadDataSet(fileName):
    num_feature = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        line_array = []
        current_line = line.strip().split('\t')
        for i in range(num_feature):
            line_array.append(float(current_line[i]))
        dataMat.append(line_array)
        labelMat.append(float(current_line[-1]))
    return dataMat, labelMat


# 计算回归系数w
def standRegres(xArr,yArr):
    """
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 根据文中推导的公示计算回归系数
    xTx = xMat.T * xMat
    # 调用linalg.det()来计算行列式
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet('./ex0.txt')
    weights = standRegres(dataMat, labelMat)
    print(weights)
