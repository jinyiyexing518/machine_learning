def classify(input_tree, feature_labels, test_vector):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feature_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vector[feature_index].name == 'dict':
            class_label = classify(second_dict[key], feature_labels, test_vector)
        else:
            class_label = second_dict[key]
    return class_label
