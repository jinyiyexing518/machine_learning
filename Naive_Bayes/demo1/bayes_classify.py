from Naive_Bayes.demo1.bayes import train_naive_bayes, load_dataSet, create_vocabulary_list, set_words_to_vector
import numpy as np


def naive_bayes_classify(vector_to_classify, p0vec, p1vec, pClass1):
    p1 = sum(vector_to_classify * p1vec) + np.log(pClass1)
    p0 = sum(vector_to_classify * p0vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def naive_bayes_testing():
    list_post, list_class = load_dataSet()
    vocabulary_list = create_vocabulary_list(list_post)
    train_mat = []
    for post in list_post:
        train_mat.append(set_words_to_vector(vocabulary_list, post))
    p0vec, p1vec, p1Class = train_naive_bayes(train_mat, list_class)

    test_entry = ['love', 'my', 'dalmation']
    test_vector = np.array(set_words_to_vector(vocabulary_list, test_entry))
    print("{} is classified as {}".format(test_entry, naive_bayes_classify(test_vector, p0vec, p1vec, p1Class)))

    test_entry = ['stupid', 'garbage']
    test_vector = np.array(set_words_to_vector(vocabulary_list, test_entry))
    print("{} is classified as {}".format(test_entry, naive_bayes_classify(test_vector, p0vec, p1vec, p1Class)))


if __name__ == "__main__":
    naive_bayes_testing()