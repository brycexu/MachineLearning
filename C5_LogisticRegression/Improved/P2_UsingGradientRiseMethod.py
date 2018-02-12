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

#随机梯度上升算法
def stocGradAscent(dataMatrix,classLabels):
    #为便于计算，转为Numpy数组
    dataMat=array(dataMatrix)
    #获取数据集的行数和列数
    m,n=shape(dataMatrix)
    #初始化权值向量各个参数为1.0
    weights=ones(n)
    #设置步长为0.01
    alpha=0.01
    #循环m次，每次选取数据集一个样本更新参数
    for i in range(m):
        #计算当前样本的sigmoid函数值
        h=sigmoid(sum(dataMat[i]*weights))
        #计算当前样本的残差(代替梯度)
        error=(classLabels[i]-h)
        #更新权值参数
        weights=weights+alpha*error*dataMat[i]
    return weights

#优化随机梯度上升算法
#@dataMatrix：数据集列表
#@classLabels：标签列表
#@numIter：迭代次数，默认150
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    #将数据集列表转为Numpy数组
    dataMat=array(dataMatrix)
    #获取数据集的行数和列数
    m,n=shape(dataMatrix)
    #初始化权值参数向量每个维度均为1
    weights=ones(n)
    #循环每次迭代次数
    for j in range(numIter):
        #获取数据集行下标列表
        dataIndex=range(m)
        #遍历行列表
        for i in range(m):
            #每次更新参数时设置动态的步长，且为保证多次迭代后对新数据仍然具有一定影响
            #添加了固定步长0.1
            #第一个改进的地方：缓解数据波动或高频运动
            alpha=4/(1.0+j+i)+0.1
            #随机获取样本
            #第二个改进的地方：缓解周期性波动
            randomIndex=int(random.uniform(0,len(dataIndex)))
            #计算当前sigmoid函数值
            h=sigmoid(sum(dataMat[randomIndex]*weights))
            #计算权值更新
            #***********************************************
            error=classLabels[randomIndex]-h
            weights=weights+alpha*error*dataMat[randomIndex]
            #***********************************************
            #选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
            del(dataIndex[randomIndex])
    return weights

