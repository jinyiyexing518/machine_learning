import numpy as np
from Logistic_Regression.demo1.logistic_regression import sigmoid
from Logistic_Regression.demo1.logistic_regression import random_gradient_ascent_modify


def classify_vector(intX, weights):
    probability = sigmoid(sum(intX * weights))
    if probability > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('./horseColicTraining.txt')
    frTest = open('./horseColicTest.txt')
    training_Set = []
    training_labels = []
    for line in frTrain.readlines():
        currentline = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(currentline[i]))
        training_Set.append(line_array)
        training_labels.append(float(currentline[21]))
    train_weights = random_gradient_ascent_modify(np.array(training_Set), training_labels, 500)
    error_count = 0
    num_test_vector = 0.0
    for line in frTest.readlines():
        num_test_vector += 1.0
        currentline = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(currentline[i]))
        classify_result = classify_vector(line_array, train_weights)
        if int(classify_result) != int(currentline[21]):
            error_count += 1
    error_rate = (float(error_count) / num_test_vector)
    print("The error rate of this test is :{}".format(error_rate))
    return error_rate


def multiTest():
    num_test = 10
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colicTest()
    print("After {} iterations the average error rate is:{}".format(num_test, error_sum/float(num_test)))


if __name__ == "__main__":
    multiTest()





