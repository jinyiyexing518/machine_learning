import numpy as np


def loadDateSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        dataMat.append([float(current_line[0]), float(current_line[1])])
        labelMat.append(float(current_line[2]))
    return dataMat, labelMat


def select_rand(i, m):
    """
    :param i: 第一个 alpha 的下标
    :param m: 所有 alpha 的数目
    """
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """
    用于调整大于 H 或者小于 L 的 alpha的值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    :param dataMatIn: 输入数据
    :param classLabels: 标签
    :param C: 常数C，用于控制松弛因子
    :param tolerance: 容错率
    :param maxIter: 最大循环次数
    :return:
    """
    # 转换为numpy的mat存储
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    # 初始化b参数，统计dataMatrix的维度
    b = 0
    m, n = np.shape(dataMatrix)
    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代matIter次
    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            # 先计算预测值，已优化过min(w，b)，并计算出了极值点，代入用来优化max(alpha)
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 优化alpha，更设定一定的容错率
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = select_rand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :] * \
                      dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 步骤5：修剪alpha_j
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("alpha_j变化太小")
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - \
                     labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - \
                     labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num,i,alphaPairsChanged))
        # 更新迭代次数
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alphas


if __name__ == "__main__":
    dataMat, labelMat = loadDateSet('./testSet.txt')
    # print(labelMat)

    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    print(b, '\n', alphas[alphas > 0])




