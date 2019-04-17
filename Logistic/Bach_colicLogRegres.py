import numpy as np
import random
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:return 1.0
    else:return 0.0
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n) #参数初始化#存储每次更新的回归系数
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01  #降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]#更新回归系数
			del(dataIndex[randIndex])#删除已经使用的样本
	return weights
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')#Strip 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列 Split() 通过指定的分隔符对字符串进行切片，并返回分割后d字符串列表。
        lineArr = []
        for i in range(21):#22个特征
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
        #数据整理到list中完毕
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount = 0;numTestVec = 0.0
    
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))#读取数据
            
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[21]):#currLine[21] 是测试集的标准答案
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is:%f"% errorRate)
    return errorRate
def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is:%f" %(numTests,errorSum/float(numTests)))
            
    
调用：
import colicLogRegres
colicLogRegres.colicTest()
colicLogRegres.multiTest()

        
