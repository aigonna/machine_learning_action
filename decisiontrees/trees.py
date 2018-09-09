#coding=utf-8
from math import log
from collections import Counter
import operator
import pickle
import copy
#import decisionTreePlot as dtPlot

def calShannonEnt(dataSet):


    # method1
    # numEntries = len(dataSet)
    # labelCounts = {}
    # for featVec in dataSet:
    #     currentLabel = featVec[-1]
    #     if currentLabel not in labelCounts.keys():
    #         labelCounts[currentLabel] = 0
    #     labelCounts[currentLabel] += 1
    #
    # shannonEnt = 0.0
    # for key in labelCounts:
    #     prob = float(labelCounts[key])/numEntries
    #     shannonEnt = prob*log(prob, 2)

    #method2
    label_count = Counter(data[-1] for data in dataSet)
    probs = [p[1]/len(dataSet) for p in label_count.items()]
    shannonEnt = sum([-p*log(p, 2) for p in probs])
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']

    return dataSet, labels


def splitDataSet(dataSet, index, value):

    #method1
    # retDataSet = []
    # for featVec in dataSet:
    #     if featVec[index] == value:
    #         reducedFeatVec = featVec[:index]
    #         reducedFeatVec.extend(featVec[index+1:])
    #         retDataSet.append(reducedFeatVec)

    #method2
    retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == index and v == value]

    return retDataSet

def chooseBestFeatureToSplit(dataSet):

    #method1
    #last row is label
    # numFeatures = len(dataSet[0]) - 1
    # baseEntropy = calShannonEnt(dataSet)
    # bestInfoGain, bestFeature = 0.0, -1
    #
    # #iterate over all  the features
    # for i in range(numFeatures):
    #
    #     #create a list of all the examples of this feature
    #     featList = [example[i] for example in dataSet]
    #
    #     #get a set of unique values
    #     uniqueVals = set(featList)
    #     newEntropy = 0.0
    #
    #     #iterate over all the unique values DataSet
    #     for value in uniqueVals:
    #         #for every unique values split dataset
    #         subDataSet = splitDataSet(dataSet, i, value)
    #         prob = len(subDataSet)/float(len(dataSet))
    #         newEntropy += prob*calShannonEnt(subDataSet)
    #
    #     infoGain = baseEntropy - newEntropy
    #     print('infoGain', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
    #     if (infoGain > bestInfoGain):
    #         bestInfoGain = infoGain
    #         bestFeature = i
    #
    #     return bestFeature

    #method2
    base_entropy = calShannonEnt(dataSet)
    best_info_gain = 0
    best_feature = -1

    #iterate over all features
    for i in range(len(dataSet[0]) - 1):

        #count current features
        feature_count = Counter([data[i] for data in dataSet])
        #sum split dataset Shannon entropy
        new_entropy = sum(feature[1] / float(len(dataSet))*calShannonEnt(splitDataSet(dataSet, i, feature[0]))
                          for feature in feature_count.items())
        #update information gain
        info_gain = base_entropy - new_entropy
        print('No.{0} feature info gain is:{1:.3f}'.format(i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
        return  best_feature


def majorityCnt(classList):
    '''
    function: choose the majority label
    :param classList: classList row set
    :return: best feature row
    '''

    #method1
    # classCount = {}
    # for vote in classList:
    #     if vote not in classCount.keys():
    #         classCount[vote] = 0
    #     classCount[vote] +=1
    #
    #     sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #
    #     return  sortedClassCount[0][0]


    #method2
    major_label = Counter(classList).most_common(1)[0]
    return major_label

def createTree(dataSet, labels):

    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):

        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #choose the best feature to get it's label
    bestFeat = chooseBestFeatureToSplit(dataSet)

    #get label name
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        #calculation the rest labels
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        print('myTree', value, myTree)
        return myTree


def classify(inputTree, featLabels, testVec):
    '''
    classify inputTree class labels
    :param inputTree: decision tree model
    :param featLabels: label's name
    :param testVec: test input data
    :return:
    '''
    #get key value of root tree
    firstStr = list(inputTree.keys())[0]

    #getting value of root tree by key
    secondDict = inputTree[firstStr]

    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, '+++', secondDict, '---', key, '>>>', valueOfFeat)

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)

    else:
        classLabel = valueOfFeat

    return  classLabel


def stopTree(inputTree, filename):

    #method1
    # fw = open(filename, 'w')
    # pickle.dump(inputTree, fw)
    # fw.close()

    #method2
    with open(filename, 'w') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

def fishTest():
    myDat, labels = createDataSet()
    print(myDat, labels)
    calShannonEnt(myDat)
    print('1-------', splitDataSet(myDat, 0, 1))
    print('0-------', splitDataSet(myDat, 0, 0))
    print(chooseBestFeatureToSplit(myDat))
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)


if __name__ == '__main__':
    fishTest()