from K_NN.demo2.get_and_display_data import file2matrix
from K_NN.demo2.norm_data import auto_norm
from K_NN.demo1.kNN import classify0
import numpy as np


def classify_person_impression():
    result_list = ['not at all', 'in small doses', 'in large doses']
    flight_miles = float(input("Please input the miles of flight per year:"))
    percent_of_game = float(input("Please input the percentage of time spent playing video games:"))
    ice_cream_liters = float(input("Please input the liters of ice cream consumed per year:"))
    matrix, labels = file2matrix("./datingTestSet2.txt")
    normal_matrix, ranges, min_values = auto_norm(matrix)
    input_matrix = np.array([flight_miles, percent_of_game, ice_cream_liters])
    normal_input_matrix = (input_matrix - min_values) / ranges
    predict_label = classify0(normal_input_matrix, normal_matrix, labels, 3)
    print("The impression is :{}".format(result_list[predict_label - 1]))


if __name__ == "__main__":
    classify_person_impression()
