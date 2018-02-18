# coding:utf8
'''
Created on 2018年2月14日
@author: XuXianda

'''
from numpy import *
#拆分数据集函数，二元拆分法    
#@dataSet：待拆分的数据集
#@feature：作为拆分点的特征索引
#@value：特征的某一取值作为分割值
def binSplitDataSet(dataSet,feature,value):
    #采用条件过滤的方法获取数据集每个样本目标特征的取值大于
    #value的样本存入mat0
    #左子集列表的第一行
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:][0]
    #同上，样本目标特征取值不大于value的样本存入mat1
    mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:][0]
    #返回获得的两个列表
    return mat0,mat1