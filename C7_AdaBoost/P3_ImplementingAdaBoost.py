# coding:utf8
'''
Created on 2018年2月12日
@author: XuXianda

'''
from numpy import *
import P2_BuildingStump
#完整AdaBoost算法实现
#算法实现伪代码
#对每次迭代：
    #利用buildStump()函数找到最佳的单层决策树
    #将最佳单层决策树加入到单层决策树数组
    #计算alpha
    #计算新的权重向量D
    #更新累计类别估计值
    #如果错误率为等于0.0，退出循环

#adaBoost算法
#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数    
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    #弱分类器相关信息列表
    weakClassArr=[]
    #获取数据集行数
    m=shape(dataArr)[0]
    #初始化权重向量的每一项值相等
    D=mat(ones((m,1))/m)
    #累计估计值向量
    aggClassEst=mat(zeros((m,1)))
    #循环迭代次数
    for i in range(numIt):
        #根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst=P2_BuildingStump.buildStump(dataArr,classLabels,D)
        #打印权重向量
        print("D:",D.T)
        #求单层决策树的系数alpha
        alpha=float(0.5*log((1.0-error)/(max(error,1e-16))))
        #存储决策树的系数alpha到字典
        bestStump['alpha']=alpha
        #将该决策树存入列表
        weakClassArr.append(bestStump)
        #打印决策树的预测结果
        print("classEst:",classEst.T)
        #预测正确为exp(-alpha),预测错误为exp(alpha)
        #即增大分类错误样本的权重，减少分类正确的数据点权重
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        #更新权值向量
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #累加当前单层决策树的加权预测值
        aggClassEst+=alpha*classEst
        print("aggClassEst",aggClassEst.T)
        #求出分类错的样本个数
        aggErrors=multiply(sign(aggClassEst)!=\
                    mat(classLabels).T,ones((m,1)))
        #计算错误率
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        #错误率为0.0退出循环
        if errorRate==0.0:break
    #返回弱分类器的组合列表
    return weakClassArr