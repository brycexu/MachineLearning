# coding:utf8
'''
Created on 2018年2月28日
@author: XuXianda

'''
import LoadingDataset
import CreatingFPTree
import FindFrequentSets
#封装：通过构建FPTree来从数据集中获取满足最小支持度条件的频繁项集
def FindFrequentSetsWithFPTree(dataSet,minSupport):
    initSet=LoadingDataset.createInitSet(dataSet)
    myFPTree,myHeaderTab=CreatingFPTree.createTree(initSet,minSupport)
    freqItems=[]
    FindFrequentSets.mineTree(myFPTree,myHeaderTab,minSupport,set([]),freqItems)
    return freqItems
#dataSet：数据集
#minSupport：最小支持度
#freqItems：频繁项集
dataSet=LoadingDataset.loadSimpDat()
minSupport=3
freqItems=FindFrequentSetsWithFPTree(dataSet,minSupport)
print freqItems