#coding = utf-8
from numpy import *
import operator
from collections import Counter
from os import listdir
import numpy as np

def classify0(inX, dataSet, labels, k):
    # method1
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    #method2
    # dist = np.sum((inX - dataSet)**2, axis=1)**0.5
    # k_labels = [labels[index] for index in dist.argsort()[0:k]]
    # label = Counter(k_labels).most_common(1)[0][0]
    # return label

def img2vetcor(filename):

    returnVect = zeros((1, 1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        #print(lineStr)
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])

    return returnVect

def handwritingClassTest():
    hwLabels = []

    #loading training set
    trainingFilelist = listdir('E:\MachineLearning-python-2.7\input\\2.KNN\\trainingDigits')
    m = len(trainingFilelist)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFilelist[i]
        fileStr = fileNameStr.split('.')[0] #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vetcor('E:\MachineLearning-python-2.7\input\\2.KNN\\trainingDigits/%s'
                                       % fileNameStr)

    #loading testing set
    testFilelist = listdir('E:\MachineLearning-python-2.7\input\\2.KNN\\testDigits')
    errorCount = 0.0
    mTest = len(testFilelist)

    for i in range(mTest):
        fileNameStr = testFilelist[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        vectorUnderTest = img2vetcor('E:\MachineLearning-python-2.7\input\\2.KNN\\testDigits/%s'
                                       % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is : %d" %(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" %errorCount)
    print("\nthe total error rate is : %f" %(errorCount / float(mTest)))

if __name__ == '__main__':
    handwritingClassTest()

