# coding:utf8
'''
Created on 2018年2月1日
@author: XuXianda

'''
def loadDataSet():
#词条切分后的文档集合，列表每一行代表一个文档
    postingList=[['my','dog','has','flea',\
                  'problems','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',
                  'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['my','licks','ate','my','steak','how',\
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    #由人工标注的每篇文档的类标签
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
