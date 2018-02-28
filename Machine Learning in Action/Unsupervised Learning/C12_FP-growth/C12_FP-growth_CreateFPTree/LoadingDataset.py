# coding:utf8
'''
Created on 2018年2月26日
@author: XuXianda

'''
def loadSimpDat():
    #数据集
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
 
def createInitSet(dataSet):
    #字典形式{数据集:数据集出现次数}
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

