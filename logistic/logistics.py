#coding= utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    '''
    description:loading and parse data
    :return: dataMat: raw data features
             labelMat:raw data labels
    '''
    dataMat=[]
    labelMat=[]
    fr = open(filename)

    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(intX):
    return 1.0/(1+exp(intX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500 #iteration num
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #h is a vector
        error = (labelMat - h) #compute the difference between real type and predict type
        weights = weights + alpha*dataMatrix.transpose()*error
    return array(weights) #return the best parameter

def plotBestFit(dataArr, labelMat, weights):
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



def stoGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n) #initialize to all ones

    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

def stoGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def simpleTest():
    dataMat, labelMat = loadDataSet('E:\ML\input\\5.Logistic\\testSet.txt')
    dataArr = array(dataMat)
    weights = stoGradAscent1(dataArr, labelMat)
    plotBestFit(dataArr, labelMat, weights)

def classifyVector(intX, weights):
    prob = sigmoid(sum(intX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    ftrain = open('E:\ML\input\\5.Logistic\horseColicTraining.txt')
    ftest = open('E:\ML\input\\5.Logistic\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in ftrain.readlines():
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(lineArr)
    trainWeights = stoGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in ftest.readlines():
        numTestVec += 1
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currline[i]))
        if int(classifyVector(array(lineArr), trainWeights))!=int(currline[21]):
            errorCount +=1
    errorRate = (float(errorCount)/numTestVec)
    print('test error rate is : %f' %errorRate)
    return errorRate
def mulTest():
    numTest = 6
    errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print("after %d iteration the average error rate is %f" %(numTest, errorSum/float(numTest)))

if __name__ == '__main__':
    #simpleTest() #result is awful, need change algorithm
    mulTest()
