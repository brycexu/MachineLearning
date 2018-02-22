# coding:utf8
'''
Created on 2018年2月21日
@author: XuXianda

'''
from numpy import *
def loadDataSet():
    #构造测试数据
    return([1,3,4],[2,3,5],[1,2,3,5],[2,5])

def createC1(dataSet):
    #构造元候选项集组成的集合
    #{1},{2},{3},{4},{5}
    C1=[]
    #transaction:每一条数据[1,3,4],...
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #frozenset:不可改变类型，相比于set，可以将这些集合作为字典键值使用
    return map(frozenset, C1)

def scanD(D,Ck,minSupport):
    #Ck:k元候选项集组成的集合
    #k=2:{1,2},{1,3},...
    #D:数据集
    #minSupport:最小支持度，用来筛选
    #scanD:从Ck中筛选出支持度大于最小支持度的k元集合
    #ssCnt:字典{'k元集合':'出现次数'}
    ssCnt={}
    #第一个循环:为每一个k元集合计数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                #在每一条数据中，如果Ck中某一个k元集合出现
                #第一次出现:value=1;不是第一次出现:value+1
                if not ssCnt.has_key(can):ssCnt[can]=1
                else: ssCnt[can]+=1
    #numItems:数据集长度
    numItems=float(len(D))
    retList=[]
    supportData={}
    #第二个循环:筛选
    for key in ssCnt:
        #support:支持度
        support=ssCnt[key]/numItems
        if support>=minSupport:
            #反向插入满足要求的k元集合们
            retList.insert(0, key)
        supportData[key]=support
    #retList:满足要求的k元集合组成的列表
    #supportData:每个k元集合和它们支持度组成的字典
    return retList,supportData