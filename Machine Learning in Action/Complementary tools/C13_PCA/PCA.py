# coding:utf8
'''
Created on 2018年3月2日
@author: XuXianda

'''
from numpy import *
def loadDataSet(filename,delim='\t'):
    #解析文本数据函数
    #delim：每一行不同特征数据之间的分隔方式，默认是tab键'\t'
    #打开文本文件
    fr=open(filename)
    #将文本中每一行的特征分隔开来
    #每一行的列表对应文本中每一行
    #行中的每一列对应各个分割开来的特征
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    #利用map()，将列表中每一行的数据值映射为float型
    datArr=[map(float,line) for line in stringArr]
    #将float型数据值的列表转化为矩阵返回
    return mat(datArr)

def pca(dataMat,topNfeat):
    #dataMat：数据集矩阵
    #topNfeat：要压缩成的维度数
    #求数据矩阵每一列的均值
    meanVals=mean(dataMat,axis=0)
    #数据矩阵每一列特征减去该列的均值
    meanRemoved=dataMat-meanVals
    #计算去均值数据矩阵的协方差矩阵
    #rowvar=0：除数是n-1；rowvar=1：除数是n
    covMat=cov(meanRemoved,rowvar=0)
    #计算去均值数据矩阵的协方差矩阵的特征值以及对应的特征向量
    eigVals,eigVects=linalg.eig(mat(covMat))
    #argsort()：对特征值矩阵由小到大排序，返回对应排序后的索引
    eigValInd=argsort(eigVals)
    #挑选最大的N个特征值
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    #将最大的N个特征值对应的特征向量提取出来，组成压缩矩阵
    redEigVects=eigVects[:,eigValInd]
    #lowDDataMat：去均值数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat=meanRemoved*redEigVects
    #利用降维后的矩阵反构出原数据矩阵，用作测试，可跟未压缩的原矩阵比对
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat
    