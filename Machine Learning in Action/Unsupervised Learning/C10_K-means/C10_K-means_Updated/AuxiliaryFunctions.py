# coding:utf8
'''
Created on 2018年2月19日
@author: XuXianda

'''
from numpy import *
#文本数据解析函数
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        #将每一行的数据映射成float型
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#数据向量计算欧式距离    
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#随机初始化K个质心(质心满足数据边界之内)
def randCent(dataSet,k):
    #得到数据样本的维度
    n=shape(dataSet)[1]
    #初始化为一个(k,n)的矩阵
    centroids=mat(zeros((k,n)))
    #遍历数据集的每一维度
    for j in range(n):
        #得到该列数据的最小值
        minJ=min(dataSet[:,j])
        #得到该列数据的范围(最大值-最小值)
        rangeJ=float(max(dataSet[:,j])-minJ)
        #k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    #返回初始化得到的k个质心向量
    return centroids