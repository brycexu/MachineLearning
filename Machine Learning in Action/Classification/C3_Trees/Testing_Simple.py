# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
import P1_CreatingDatasetAndLabels
import P2_CreatingTheTree
import P3_TestingTheTree
import Extra_PlottingTheTree
myDataset,labels=P1_CreatingDatasetAndLabels.creatDataSet()
myTree=P2_CreatingTheTree.createTree(myDataset, labels)
print(P3_TestingTheTree.classify(myTree, labels, [1,0]))
print(P3_TestingTheTree.classify(myTree,labels,[1,1]))
Extra_PlottingTheTree.createPlot(myTree)