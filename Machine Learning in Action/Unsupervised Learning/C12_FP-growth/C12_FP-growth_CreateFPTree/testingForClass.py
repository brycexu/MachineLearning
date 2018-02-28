# coding:utf8
'''
Created on 2018年2月26日
@author: XuXianda

'''
import Class
rootNode=Class.treeNode('root',9,None)
rootNode.children['boy']=Class.treeNode('boy',13,None)
rootNode.children['daughter']=Class.treeNode('daughter',3,None)
rootNode.children['daughter'].children['grandchild']=Class.treeNode('grandchild',7,None)
rootNode.disp()
