# coding:utf8
'''
Created on 2018年2月15日
@author: XuXianda

'''
from numpy import *
import P1_CreatingDataset
import P4_CreatingRegressionTree
import P5_PruningTree
#看看预剪枝的效果
myDat2=P1_CreatingDataset.loadDatabase('ex2.txt')
myMat2=mat(myDat2)
print P4_CreatingRegressionTree.createTree(myMat2)
#看看后剪枝的效果
#创建所有可能中最大的树
myTree=P4_CreatingRegressionTree.createTree(myMat2,ops=(0,1))
myDatTest=P1_CreatingDataset.loadDatabase('ex2test.txt')
myMatTest=mat(myDatTest)
print P5_PruningTree.prune(myTree, myMatTest)
#其实，后剪枝的效果没有预剪枝的效果好，所以，一般来说是结合使用的