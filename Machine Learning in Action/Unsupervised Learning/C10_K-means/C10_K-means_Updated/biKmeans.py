# coding:utf8
'''
Created on 2018年2月20日
@author: XuXianda

'''
import AuxiliaryFunctions
import KMeans
from numpy import *
#二分K-均值聚类算法
#@dataSet:待聚类数据集
#@k：用户指定的聚类个数
#@distMeas:用户指定的距离计算方法，默认为欧式距离计算
def biKmeans(dataSet,k,distMeas=AuxiliaryFunctions.distEclud):
    #获得数据集的样本数
    m=shape(dataSet)[0]
    #初始化一个元素均值0的(m,2)矩阵
    clusterAssment=mat(zeros((m,2)))
    #获取数据集每一列数据的均值，组成一个长为列数的列表
    centroid0=mean(dataSet,axis=0).tolist()[0]
    #当前聚类列表为将数据集聚为一类
    centList=[centroid0]
    #遍历每个数据集样本
    for j in range(m):
        #计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    #循环，直至二分k-均值达到k类为止
    while (len(centList)<k):
        #将当前最小平方误差置为正无穷
        lowerSSE=inf
        #遍历当前每个聚类
        for i in range(len(centList)):
            #通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat,splitClustAss=KMeans.kMeans(ptsInCurrCluster,2,distMeas)
            #计算该类划分后两个类的误差平方和
            sseSplit=sum(splitClustAss[:,1])
            #计算数据集中不属于该类的数据的误差平方和
            sseNotSplit=\
                sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #打印这两项误差值
            print 'sseSplit,and notSplit:',(sseSplit,sseNotSplit)
            #划分第i类后总误差小于当前最小总误差
            if(sseSplit+sseNotSplit)<lowerSSE:
                #第i类作为本次划分类
                bestCentToSplit=i
                #第i类划分后得到的两个质心向量
                bestNewCents=centroidMat
                #复制第i类中数据点的聚类结果即误差值
                bestClustAss=splitClustAss.copy()
                #将划分第i类后的总误差作为当前最小误差
                lowerSSE=sseSplit+sseNotSplit
        #数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
        #当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        #同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        #连续不出现空缺
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        #打印本次执行2-均值聚类算法的类
        print 'the bestCentToSplit is:',bestCentToSplit
        #打印被划分的类的数据个数
        print 'the len of bestClustAss is:',(len(bestClustAss))
        #更新质心列表中的变化后的质心向量
        centList[bestCentToSplit]=bestNewCents[0,:].tolist()[0]
        #添加新的类的质心向量
        centList.append(bestNewCents[1,:].tolist()[0])
        #更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
        #返回聚类结果
    return mat(centList),clusterAssment
