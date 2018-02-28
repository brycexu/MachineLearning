# coding:utf8
'''
Created on 2018年2月26日
@author: XuXianda

'''
import Class
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 子树末端有该元素项时计数值+1
        inTree.children[items[0]].inc(count)
    else:
        # 子树末端没有这个元素项时创建一个新节点
        inTree.children[items[0]] = Class.treeNode(items[0], count, inTree)
        # 更新元素指针
        if headerTable[items[0]][1] == None:
            # 从头指针表指向第一个元素节点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 从前一个元素节点指向新节点
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 对剩下的元素项迭代调用updateTree函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        # 追溯到最头头
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
