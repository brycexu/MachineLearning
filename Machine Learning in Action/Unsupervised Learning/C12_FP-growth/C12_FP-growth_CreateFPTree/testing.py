# coding:utf8
'''
Created on 2018年2月26日
@author: XuXianda

'''
import LoadingDataset
import CreatingFPTree
simpDat=LoadingDataset.loadSimpDat()
initSet=LoadingDataset.createInitSet(simpDat)
print initSet
myFPtree,myHeaderTab=CreatingFPTree.createTree(initSet,3)
myFPtree.disp()
