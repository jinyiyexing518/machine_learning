from Decision_Tree.demo1.create_data import create_dataSet


def split_dataSet(dataSet, axis, values):
    return_dataSet = []
    for feature_vector in dataSet:
        if feature_vector[axis] == values:
            # 划分后，把axis维度的特征除外
            reduceFeatVec = feature_vector[:axis]
            reduceFeatVec.extend(feature_vector[axis+1:])
            return_dataSet.append(reduceFeatVec)
    return return_dataSet


if __name__ == "__main__":
    myData, labels = create_dataSet()
    dataset = split_dataSet(myData, 0, 1)
    print(dataset)
