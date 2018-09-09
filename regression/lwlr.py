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

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#eye返回对角阵

    for j in range(m):
        #testpoint的形式是一个行向量,计算testpoint与输入样本的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - xMat[j, :]
        #k控制衰减速度 w[i] = exp(-(x[i]-x)/(2*k^2))
        weights[j, j] = exp(diffMat * diffMat.T/(-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))#（theater = (X'*X)^-1*X’*(w[i]*y)）
    return testPoint * ws



def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def regression():
    xArr, yArr = loadDataSet('D:\project\machine_learning_action\data\8.Regression\data.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    strInd = xMat[:, 1].argsort(0)
    xSort = xMat[strInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1].flatten().A[0], mat(yHat[strInd]).T.flatten().A[0])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def abaloneTest():
    abX, abY = loadDataSet('D:\project\machine_learning_action\data\8.Regression\\abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print(rssError(abY[0:99], yHat01.T))
    print(rssError(abY[0:99], yHat1.T))
    print(rssError(abY[0:99], yHat10.T))

def ridgeRegres(xMat, yMat, lam = 0.2):
    """
    岭回归
    :param xMat:
    :param yMat:
    :param lam: 调节系数
    :return:
    """
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("矩阵奇异,不可计算")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


if __name__ == "__main__":
    regression()
    #abaloneTest()