# coding:utf8
'''
Created on 2018年2月14日
@author: XuXianda

'''
from numpy import *
import P1_CreatingDataset
import P4_CreatingRegressionTree
myDat=P1_CreatingDataset.loadDatabase('ex00.txt')
myMat=mat(myDat)
print('The Regression Tree for ex00:')
print P4_CreatingRegressionTree.createTree(myMat)
myDat1=P1_CreatingDataset.loadDatabase('ex0.txt')
myMat1=mat(myDat1)
print('The Regression Tree for ex0:')
print P4_CreatingRegressionTree.createTree(myMat1)
