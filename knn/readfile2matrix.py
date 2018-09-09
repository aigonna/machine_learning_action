from numpy import *

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



datingDataMat, datingLabels = file2matrix(
        'D:\project\machine_learning_action\data\\2.KNN\datingTestSet2.txt')
print(datingDataMat, datingLabels)



