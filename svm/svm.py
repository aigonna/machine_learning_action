#codig = utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):

    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    '''
    function: generate a num j
    :param i: first alpha target
    :param m: all alpha nums
    :return:  return not i, between 0 and m random numbers
    '''
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    '''
    function;  tune aj(L<=aj<=H)
    :param aj: target value
    :param H:  max value
    :param L:  min value
    :return:   aj-> target value
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    b = 0
    alphas = mat(zeros((m, 1)))

    iter = 0
    while (iter<maxIter):
        alphaParisChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])

            if((labelMat[i]*Ei < toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaI_old = alphas[i].copy()
                alphaJ_old = alphas[j].copy()
                if (labelMat[i])!=labelMat[j]:
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H:
                    print('L=H')
                    continue
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T-dataMatrix[i, :]*dataMatrix[i, :].T-dataMatrix[j, :]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if (abs(alphas[j]-alphaJ_old) < 0.000001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[i]*labelMat[i]*(alphaJ_old-alphaI_old)
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaI_old)*dataMatrix[i, :]*dataMatrix[i, :].T-labelMat[j]*(alphas[j]-alphaJ_old)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaI_old)*dataMatrix[i, :]*dataMatrix[j, :].T-labelMat[j]*(alphas[j]-alphaJ_old)*dataMatrix[j,:]*dataMatrix[j,:].T

                if (0<alphas[i]) and (C>alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaParisChanged += 1
                print("iter: %d i:%d, pairs changed %d " %(iter, i, alphaParisChanged))
        if (alphaParisChanged==0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" %iter)
    return b, alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w

def plotfig_SVM(xMat, yMat, ws, b, alphas):
    xMat = mat(xMat)
    yMat = mat(yMat)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])
    x = arange(-1.0, 10.0, 0.1)
    y = (-b - ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
    for i in range(100):
        if alphas[i]>0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('D:\project\machine_learning_action\data\\6.SVM\\testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    ws = calcWs(alphas, dataArr, labelArr)
    plotfig_SVM(dataArr, labelArr, ws, b, alphas)
