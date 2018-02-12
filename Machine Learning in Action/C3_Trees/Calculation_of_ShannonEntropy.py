# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
#计算给定数据集的熵
#导入log运算符
from math import log

def calEnt(dataSet):
    #获取数据集的行数
    numEntries=len(dataSet)
    #设置字典的数据结构
    labelCounts={}
    #提取数据集的每一行的特征向量
    for featVec in dataSet:
        #获取特征向量的最后一列的标签
        currentLabel=featVec[-1]
        #检测字典的关键字key中是否存在该标签
        #如果不存在keys()关键字
        if currentLabel not in labelCounts.keys():
            #将当前标签/0键值对存入字典中
            labelCounts[currentLabel]=0
        #否则将当前标签对应的键值加1
        labelCounts[currentLabel]+=1
    #初始化熵为0
    Ent=0.0
    #对于数据集中所有的分类类别
    for key in labelCounts:
        #计算各个类别出现的频率
        prob=float(labelCounts[key])/numEntries
        #计算各个类别信息期望值
        Ent-=prob*log(prob,2)
    #返回熵
    return Ent
