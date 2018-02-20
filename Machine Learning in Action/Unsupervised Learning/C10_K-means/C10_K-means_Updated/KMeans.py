# coding:utf8
'''
Created on 2018年2月19日
@author: XuXianda

'''
from numpy import *
import AuxiliaryFunctions
#k-均值聚类算法
#@dataSet:聚类数据集
#@k:用户指定的k个类
#@distMeas:距离计算方法，默认欧氏距离distEclud()
#@createCent:获得k个质心的方法，默认随机获取randCent()
def kMeans(dataSet,k,distMeas=AuxiliaryFunctions.distEclud,createCent=AuxiliaryFunctions.randCent):
    #获取数据集样本数
    m=shape(dataSet)[0]
    #初始化一个(m,2)的矩阵
    clusterAssment=mat(zeros((m,2)))
    #创建初始的k个质心向量
    centroids=createCent(dataSet,k)
    #聚类结果是否发生变化的布尔类型
    clusterChanged=True
    #只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化
    while clusterChanged:
        #聚类结果变化布尔类型置为false
        clusterChanged=False
        #遍历数据集每一个样本向量
        for i in range(m):
            #初始化最小距离最正无穷；最小距离对应索引为-1
            minDist=inf;minIndex=-1
            #循环k个类的质心
            for j in range(k):
                #计算数据点到质心的欧氏距离
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                 #如果距离小于当前最小距离
                if distJI<minDist:
                    #当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)
                     minDist=distJI;minIndex=j
             #当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
            if clusterAssment[i,0] !=minIndex:clusterChanged=True
             #更新当前变化样本的聚类结果和平方误差
            clusterAssment[i,:]=minIndex,minDist**2
         #打印k-均值聚类的质心
        #print centroids
         #遍历每一个质心
        for cent in range(k):
            #将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
             #计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[cent,:]=mean(ptsInClust,axis=0)
    #返回k个聚类，聚类结果及误差
    return centroids,clusterAssment
