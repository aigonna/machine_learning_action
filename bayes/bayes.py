#coding=utf-8
from numpy import *
import re
import operator
import feedparser

def loadDataSet():
    '''
    create a dataset
    :return:
    word list:postinglist
    class: classVec
    '''

    postinglist = [
        ['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
        ['maybe', 'not', 'take', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'cute', 'I', 'love', 'hime'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']

    ]
    classVec = [0, 1, 0, 1, 0, 1] #1 is abusing, 0 not
    return postinglist, classVec


def createVocalList(dataSet):
    '''
    function: get all vocabulary set
    :param dataSet: dataset
    :return: vocabulary set
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document) #union of the two sets
    return list(vocabSet)


def setOfwords2Vec(vocabList, inputSet):
    '''
    function:iterate all words, if it appeared, set word to 1
    :param vocabList: all words set list
    :param inputSet: input dataset
    :return: returnVec : matching list
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary! " %word)
    return returnVec

def bagOfWods2VecMN(vocabList, inputSet):
    '''
    function:count word appear times
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word : %s is not in my vocabulary!" %word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    function: calculate normal or abused word appear frenquency
    :param trainMatrix: in class 0, the proportion of every word appear nums
    :param trainCategory: in class 1, the proportion of every word appear nums
    :return: p0Vect: in class 0, the proportion of every word appear nums
             p1Vect: in class 1, the proportion of every word appear nums
             pAbusive:probability of abusive word
    '''

    #numbers of all files
    numTrainDocs = len(trainMatrix)

    #numbers of all words
    numWords = len(trainMatrix[0])

    #the probability of abusive word appear
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #create a list of normal or abused word appear numbers
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    #count normal or abused word
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])

        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    function: classify pre test data to abused or normal
    :param vec2Classify:pre test data-> pre classify vector
    :param p0Vec: normal files
    :param p1Vec: abusive files
    :param pClass1: probabilit of abusive files appear
    :return: class 1 or 0
    '''

    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0



def testNB():
    list0Posts, listClasses = loadDataSet()
    myVocabList = createVocalList(list0Posts)
    trainMat = []

    for postindoc in list0Posts:
        trainMat.append(setOfwords2Vec(myVocabList, postindoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfwords2Vec(myVocabList, testEntry))
    print(testEntry, 'classify as:', classifyNB(thisDoc, p0V, p1V, pAb))

#********************************************************************************************************
def textParse(bigString):
    '''
    function:return big string to a lower string list
    :param bigString: input upper characters
    :return:lower string list
    '''
    listOfTokens = re.split('x*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamtest():

    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):

        wordList = textParse(open('E:\ML\input\\4.NaiveBayes\email\spam\%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)
        classList.append(1)

        wordList = textParse(open('E:\ML\input\\4.NaiveBayes\email\ham\%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocalList(docList)
    trainingSet = list(range(50))
    testSet = []

    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfwords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])


    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfwords2Vec(vocabList, docList[docIndex])

        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error count is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount)/len(testSet))


def testParsetext():
    print(textParse(open('E:\ML\input\\4.NaiveBayes\email\spam\\1.txt', "rb").read().decode('GBK', 'ignore')))


#**********************************************************************************************************

def calcMostFreq(vocabList, fullText):

    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)


    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):

    docList = []
    classList = []
    fullText = []

    minlen=min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minlen):

        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocalList(docList)
    top30Words = calcMostFreq(vocabList, fullText)

    for pairw in top30Words:
        if pairw[0] in vocabList:
            vocabList.remove(pairw[0])


    trainingSet = list(range(2*minlen))
    testSet = []

    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(bagOfWods2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWods2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam)!=classList[docIndex]:
            errorCount +=1
    print('the error rate is :', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):


    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print("SF****************************************SF")
    for item in sortedSF:
        print(item[0])

    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print("NY*****************************************NY")
    for item in sortedNY:
        print(item[1])


if __name__ == '__main__':
    #testNB()
    #spamtest()
    #testParsetext()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)


