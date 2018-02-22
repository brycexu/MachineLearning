# coding:utf8
'''
Created on 2018年2月21日
@author: XuXianda

'''
from numpy import *
import AuxiliaryFunctions
def aprioriGen(Lk, k): 
    #创建C(k+1),即输入的k元集合所能组成的所有k+1元集合
    #Lk:输入的由k元集合组成的列表
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            #对于第i个和第j个k元集合（j>i）
            #思想:如果排序后前k-2个元素相同，那么它们就可以组成一个新的k+1元集合
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j]) 
    #retList:输入的k元集合所能组成的所有k+1元集合
    return retList

def apriori(dataSet, minSupport = 0.5):
    #主算法
    #从数据集中挖掘出频繁集
    C1 = AuxiliaryFunctions.createC1(dataSet)#创建C1:元候选项集
    D = map(set, dataSet)#将数据集变为set格式
    L1, supportData = AuxiliaryFunctions.scanD(D, C1, minSupport)#L1:C1中所有符合条件的集合组成的集合
    L = [L1]
    k = 2
    #当L中某一个维度下为空时停止
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)#由Lk组成C(k+1)，即Lk中的k元集合所能组成的所有k+1元集合
        Lk, supK = AuxiliaryFunctions.scanD(D, Ck, minSupport)#再从C(k+1)中筛选所有符合条件的k+1元集合，组成L(k+1)
        supportData.update(supK)
        #添加k+1维度下的集合
        L.append(Lk)
        k += 1
    #L:所有支持度大于最小支持度的集合（频繁集）
    #supportData:记录它们支持度的字典
    return L, supportData
