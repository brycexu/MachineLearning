# coding:utf8
'''
Created on 2018年2月2日
@author: XuXianda

'''
#利用梯度上升法找到最优参数w
from numpy import *

#定义sigmoid函数
def sigmoid(inx):
    return 1.0/(1+exp(-inx))

#梯度上升法更新最优拟合参数
#@dataMatIn：数据集
#@classLabels：数据标签
def gradAscent(dataMatIn,classLabels):
    #将数据集列表转为Numpy矩阵
    dataMatrix=mat(dataMatIn)
    #将数据集标签列表转为Numpy矩阵，并转置
    labelMat=mat(classLabels).transpose()
    #获取数据集矩阵的行数和列数
    m,n=shape(dataMatrix)
    #学习步长
    alpha=0.001
    #最大迭代次数
    maxCycles=500
    #初始化权值参数向量每个维度均为1.0
    weights=ones((n,1))
    #循环迭代次数
    for k in range(maxCycles):
        ######我们可以通过求导验证logistic回归函数对参数w的梯度为(yi-sigmoid(wTx))*x
        #求当前的sigmoid函数预测概率
        h=sigmoid(dataMatrix*weights)
        #以下是在计算真实类别与预测类别的差值，然后按照差值的方向调整回归系数
        #***********************************************
        #此处计算真实类别和预测类别的差值
        #对logistic回归函数的对数释然函数的参数项求偏导
        error=(labelMat-h)
        #更新权值参数                                                     
        weights=weights+alpha*dataMatrix.transpose()*error
        #***********************************************
    return weights
