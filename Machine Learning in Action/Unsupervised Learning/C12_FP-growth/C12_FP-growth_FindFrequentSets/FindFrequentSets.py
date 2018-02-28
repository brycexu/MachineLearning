# coding:utf8
'''
Created on 2018年2月28日
@author: XuXianda

'''
import CreatingFPTree
def findPrefixPath(basePat, treeNode):
    #创建某元素的条件模式基（前缀路径）
    #basePat：某元素
    #treeNode：HeaderTab指针指向的第一个该元素节点
    #condPats：储存条件模式基（前缀路径）
    condPats = {}
    #treeNode：从HeaderTab的该元素指针开始，遍历该单链表上所有该元素的节点
    while treeNode != None:
        #prefixPath：记录当前元素节点上的前缀路径
        prefixPath = []
        #ascendTree：添加所有该元素节点-根节点中的节点
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            #condPats是一个字典 {前缀路径：对应元素节点的计数值}
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        #到下一个元素节点
        treeNode = treeNode.nodeLink
    return condPats

def ascendTree(leafNode, prefixPath):
    #添加所有该元素节点-根节点中的节点
    #leafNode：该前缀路径对应的元素节点
    #prefixPath：记录该前缀路径的列表
    #迭代
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #挖掘频繁项集
    #inTree和headerTable是createTree()函数生成的数据集的FPTree和HeaderTab
    #minSup：最小支持度
    #preFix传入一个空集合(set([]))，用来在函数中保存当前前缀
    #freqItemList传入一个空列表([])，用来在函数中储存生成的频繁项集
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    #将headerTable中元素按照计数值由小到大排列
    for basePat in bigL:
        #记basePat+preFix为当前频繁项集的newFreqSet
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #将newFreqSet添加到freqItemList中
        freqItemList.append(newFreqSet)
        #挖掘该元素的前缀路径
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #获得该元素的条件FPTree
        myCondTree, myHead = CreatingFPTree.createTree(condPattBases, minSup)
        #当条件FPTree不为空时，继续下一步，否则退出递归
        if myHead != None:
            # 用于测试
            print 'conditional tree for:', newFreqSet
            myCondTree.disp()
            #以条件树myCondTree和条件树头指针表myHead为新的输入，以newFreqSet为新的preFix，外加freqItemList，递归这一个过程
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
