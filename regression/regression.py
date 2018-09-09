from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    #获取样本特征总数,不算最后的目标变量
    numFeatures = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():#读取每一行
        lineArr = []
        #删除一行中以tab分割的数据前后的空白符号
        curLine = line.strip().split('\t')
        for i in range(numFeatures):#i从0到2 不包含2
            #讲数据添加到lineArr List中，每一行数据测试数据组成一个行向量
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)#讲测试数据的输入数据部分储存到DataMat的list
        labelMat.append(float(curLine[-1]))#将每一行的最后一个数据,即类别，叫目标变量储存到labelMat中
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr)#转换成mat
    yMat = mat(yArr).T#mat的转置操作

    xTx = xMat.T*xMat #X'*X
    #linalg.det求矩阵行列式,如果矩阵行列式为0,则这个矩阵是不可逆的
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    #最小二乘法
    ws = xTx.I*(xMat.T*yMat) #（theater = (X'*X)^-1*X’*y）
    return ws


def regression():
    xArr, yArr = loadDataSet('D:\project\machine_learning_action\data\8.Regression\data.txt')
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0], s=4, c='red')
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

regression()