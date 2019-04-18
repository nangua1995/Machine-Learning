from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T*xMat 
    if linalg.det(xTx) == 0.0
        print ("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat) #I 是矩阵的逆
    return ws


#绘制图像
#ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
#x[m,n]是通过numpy库引用数组或矩阵中的某一段数据集的一种写法
#x[:,n]或者x[n,:]
#x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据
#举例说明：
#import numpy as np  
  
#X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])  
#print X[:,0]  
#输出结果是： 0  2 4 6 8 10 12 14 16 18

