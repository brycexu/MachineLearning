# coding:utf8
'''
Created on 2018年2月15日
@author: XuXianda

'''
#后剪枝
from numpy import *
import P3_SplitDataset
#根据目标数据的存储类型是否为字典型，是返回true，否则返回false
def isTree(obj):
    return (type(obj).__name__=='dict')

#获取均值函数    
def getMean(tree):
    #树字典的右分支为字典类型：递归获得右子树的均值
    if isTree(tree['right']):tree['right']=getMean(tree['right'])
    #树字典的左分支为字典类型：递归获得左子树的均值
    if isTree(tree['left']):tree['left']=getMean(tree['left'])
    #递归直至找到两个叶节点，求二者的均值返回
    return (tree['left']+tree['right'])/2.0

#剪枝函数
#@tree:树字典    
#@testData:用于剪枝的测试集
def prune(tree,testData):
    #测试集为空，直接对树相邻叶子结点进行求均值操作
    if shape(testData)[0]==0:return getMean(tree)
    #左右分支中有非叶子结点类型
    if (isTree(tree['right']) or isTree(tree['left'])):
        #利用当前树的最佳切分点和特征值对测试集进行树构建过程
        lSet,rSet=P3_SplitDataset.binSplitDataSet(testData,tree['spInd'],tree['spval'])
    #左分支非叶子结点，递归利用测试数据的左子集对做分支剪枝
    if isTree(tree['left']):tree['left']=prune(tree['left'],lSet)
    #同理，右分支非叶子结点，递归利用测试数据的右子集对做分支剪枝
    if isTree(tree['right']):tree['right']=prune(tree['right'],lSet)
    #左右分支都是叶节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        #利用该子树对应的切分点对测试数据进行切分(树构建)
        lSet,rSet=P3_SplitDataset.binSplitDataSet(testData,tree['spInd'],tree['spval'])
        #如果这两个叶节点不合并，计算误差，即（实际值-预测值）的平方和
        errorNoMerge=sum(power(lSet[:,-1]-tree['left'],2))+\
                     sum(power(rSet[:,-1]-tree['right'],2))
        #求两个叶结点值的均值
        treeMean=(tree['left']+tree['right'])/2.0
        #如果两个叶节点合并，计算合并后误差,即(真实值-合并后值）平方和
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        #合并后误差小于合并前误差
        if errorMerge<errorNoMerge:
            #和并两个叶节点，返回合并后节点值
            print('merging')
            return treeMean
        #否则不合并，返回该子树
        else:return tree
    #不合并，直接返回树
    else:return tree
