import numpy as np


def img2array(filename):
    """
    将某一txt格式图像转化为一维数组
    :param filename: txt图像
    :return: 一位数组
    """
    return_array = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        # readline()每次读出一行内容，所以，读取时占用内存小，比较适合大文件
        # 返回字符串对象，每次读取时遇到换行符\n就停止，下一次从下一行继续读取
        line_str = fr.readline()
        for j in range(32):
            return_array[0, 32*i + j] = int(line_str[j])
    return return_array
