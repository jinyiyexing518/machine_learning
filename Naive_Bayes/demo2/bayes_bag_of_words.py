from Naive_Bayes.demo1.bayes import create_vocabulary_list, set_words_to_vector, train_naive_bayes
from Naive_Bayes.demo1.bayes_classify import naive_bayes_classify
import numpy as np


def bag_of_words_2vector(vocabulary_list, inputSet):
    return_vector = [0] * len(vocabulary_list)
    for word in inputSet:
        return_vector[vocabulary_list.index[word]] += 1
    return return_vector


# 这个程序实现的功能是对字符串进行处理，但是还没处理好，需要继续修改
def textParse(bigString):
    import jieba
    list_of_tokens = jieba.lcut(bigString)
    returns = [tok.lower() for tok in list_of_tokens if len(tok) > 2]
    return returns


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabularyList = create_vocabulary_list(docList)
    trainingSet = list(np.arange(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(set_words_to_vector(vocabularyList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = train_naive_bayes(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = set_words_to_vector(vocabularyList, docList[docIndex])
        if naive_bayes_classify(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("The error rate is {}".format(float(errorCount) / len(testSet)))


if __name__ == "__main__":
    spamTest()










