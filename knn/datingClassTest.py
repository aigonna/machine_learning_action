#coding = utf-8
from numpy import *
from collections import Counter
import operator
import numpy as np

def classify0(inX, dataSet, labels, k):
    # method1
    # dataSetSize = dataSet.shape[0]
    # diffMat = tile(inX, (dataSetSize, 1))-dataSet
    # sqDiffMat = diffMat**2
    # sqDistances = sqDiffMat.sum(axis=1)
    # distances = sqDistances**0.5
    # sortedDistIndicies = distances.argsort()
    #
    # classCount = {}
    # for i in range(k):
    #     voteIlabel = labels[sortedDistIndicies[i]]
    #     classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # return sortedClassCount[0][0]

    #method2
    dist = np.sum((inX - dataSet)**2, axis=1)**0.5
    k_labels = [labels[index] for index in dist.argsort()[0:k]]
    label = Counter(k_labels).most_common(1)[0][0]
    return label


def file2matrix(filename):
    fr = open(filename)

    numberoflines = len(fr.readlines())
    #print(numberoflines)
    returnMat = zeros((numberoflines, 3))
    #print(returnMat,'\n', returnMat.shape)
    classLabelVetcor = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        #print('line:\n', line)
        listFromLine = line.split('\t')
        #print('lineFromLine:\n', listFromLine)
        returnMat[index, :] = listFromLine[0:3]
        #print('returnMat:\n', returnMat)
        classLabelVetcor.append(int(listFromLine[-1]))

        index +=1
    return returnMat, classLabelVetcor

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    #method1
    # normDataSet = zeros(shape(dataSet))
    # m = dataSet.shape[0]
    # normDataSet = dataSet- tile(minVals, (m, 1))
    # normDataSet = normDataSet/tile(ranges, (m, 1))

    #method2
    normDataSet = (dataSet - minVals) / ranges

    return normDataSet, ranges, minVals


def datingClassTest():

    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix(
        'D:\project\machine_learning_action\data\\2.KNN\datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    print('numTestVecs=', numTestVecs)
    errCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
        #                              datingLabels[numTestVecs, :], 3)
        print("the classifier came back with: %d, the real answer is: %d"
               %(classifierResult, datingLabels[i]))
        if (classifierResult !=datingLabels[i]):
            errCount += 1.0
    print("the total error rate is: %f" %(errCount/float(numTestVecs)))
    print(errCount)

if __name__ == '__main__':
    datingClassTest()