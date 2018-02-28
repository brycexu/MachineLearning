# coding:utf8
'''
Created on 2018年2月26日
@author: XuXianda

'''
import AuxiliaryFunctions
import Class
def createTree(dataSet, minSup=1):
    ''' 创建FP树 '''
    # 第一次遍历数据集，创建头指针表headerTable{元素:出现次数}
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    # freItemSet:存放所有元素的集合
    freqItemSet = set(headerTable.keys())
    # 空元素集，返回空
    if len(freqItemSet) == 0:
        return None, None
    # 增加一个数据项，用于存放指向相似元素项指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 创建根节点
    retTree = Class.treeNode('Null Set', 1, None) 
    # 第二次遍历数据集，创建FP树
    # 这里的dataSet是LoadingDataset中的retDict字典
    for tranSet, count in dataSet.items():
        localD = {} # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0] # 注意这个[0]，因为之前加过一个数据项
        if len(localD) > 0:
            # 将每个项集中的元素按照全局频率排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)] # 排序
            AuxiliaryFunctions.updateTree(orderedItems, retTree, headerTable, count) # 更新FP树
    return retTree, headerTable
