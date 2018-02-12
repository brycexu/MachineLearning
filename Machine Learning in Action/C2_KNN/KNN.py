# coding:utf8
'''
Created on 2018年1月26日
@author: XuXianda

'''
#-------------------------1 准备数据-------------------------------
#可以采用公开的数据集，也可以利用网络爬虫从网站上抽取数据，方式不限
#-------------------------2 准备数据------------------------------- 
#确保数据格式符合要求
#导入科学计算包（数组和矩阵）
from numpy import *
from os import listdir
#导入运算符模块
import operator

#创建符合python格式的数据集
def createDataSet():
    #数据集list（列表格式）
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    #标签
    labels=['A','A','B','B']
    return group,labels

#-------------------------构建分类器------------------------------- 
#KNN算法实施
#@inX  测试样本数据
#@dataSet  训练样本数据
#@labels  测试样本标签
#@k  选取距离最近的k个点
def classify0(inX,dataSet,labels,k):
    #获取训练数据集的行数
    dataSetSize=dataSet.shape[0]
    #---------------欧氏距离计算-----------------
    #各个函数均是以矩阵形式保存
    #tile():inX沿各个维度的复制次数
    diffMat=tile(inX, (dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    #.sum()运行加函数，参数axis=1表示矩阵每一行的各个值相加和
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #--------------------------------------------
    #获取排序（由小到大）后的距离值的索引（序号）
    sortedDistIndicies=distances.argsort()
    #字典，键值对，结构类似于hash表
    classCount={}
    for i in range(k):
        #获取该索引对应的训练样本的标签
        voteIlabel=labels[sortedDistIndicies[i]]
        #累加几类标签出现的次数，构成键值对key/values并存于classCount中
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #将字典列表中按照第二列，也就是次数标签，反序排序（由大到小排序）
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #返回第一个元素（最高频率）标签key
    return sortedClassCount[0][0]





