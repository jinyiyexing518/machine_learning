from K_NN.demo2.get_and_display_data import file2matrix
from K_NN.demo2.norm_data import auto_norm
from K_NN.demo1.kNN import classify0


def classify_test():
    test_data_ratio = 0.1
    matrix, labels = file2matrix("./datingTestSet2.txt")
    normal_data, ranges, min_value = auto_norm(matrix)
    m = matrix.shape[0]
    test_data_num = int(test_data_ratio * m)
    test_error = 0.0
    flag = False
    for i in range(test_data_num):
        classify_result = classify0(normal_data[i], normal_data[test_data_num:], labels[test_data_num:], 3)
        if classify_result != labels[i]:
            test_error += 1.0
            flag = False
        else:
            flag = True
        print("预测分类为：{} 实际label：{} 正误：{}".format(classify_result, labels[i], flag))
    print("测试误差为：{}".format(test_error / float(test_data_num)))


if __name__ == "__main__":
    classify_test()

