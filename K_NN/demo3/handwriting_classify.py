import os
import numpy as np
from K_NN.demo3.txt_img_to_array import img2array
from K_NN.demo1.kNN import classify0


def handwriting_classify():
    handwriting_labels = []
    training_file_list = os.listdir("./trainingDigits")
    m = len(training_file_list)
    training_matrix = np.zeros((m, 1024))
    for i in range(m):
        file_name = training_file_list[i]
        # 获取当前地址
        paths = os.getcwd()
        file_path = os.path.join(paths, "trainingDigits", file_name)
        # 获取txt图像一维数组
        training_matrix[i] = img2array(file_path)
        # 获取标签
        file_label = int(file_name.split('.')[0].split('_')[0])
        handwriting_labels.append(file_label)

    test_file_list = os.listdir("./testDigits")
    error_rate = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name = test_file_list[i]
        paths = os.getcwd()
        file_path = os.path.join(paths, "testDigits", file_name)
        test_array = img2array(file_path)
        test_truth_label = int(file_name.split('.')[0].split('_')[0])
        classify_result = classify0(test_array, training_matrix, handwriting_labels, 3)
        if classify_result != test_truth_label:
            error_rate += 1.0
            flag = False
        else:
            flag = True
        print("The predict result is {} ; The real result is {} Result is {}".format(classify_result, test_truth_label, flag))
    print("The test error rate is {}".format(error_rate/float(m_test)))


if __name__ == "__main__":
    handwriting_classify()

