# coding:utf8
'''
Created on 2018年2月10日
@author: XuXianda

'''
from numpy import *
import AuxiliaryFunctions
import SMO_Updated
dataArr,labelArr=AuxiliaryFunctions.loadDataSet("testSet.txt")
b,alphas=SMO_Updated.smoP(dataArr, labelArr, 0.6, 0.001, 40)
#求出了alpha值和对应的b值，就可以求出对应的w值，以及分类函数值
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr);labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
ws=calcWs(alphas, dataArr, labelArr)
dataMat=mat(dataArr)
print dataMat[0]*mat(ws)+b
print labelArr[0]

