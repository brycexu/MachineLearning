# coding:utf8
'''
Created on 2018年2月12日
@author: XuXianda

'''
from numpy import *
#获取数据集
def loadSimpData():
    dataMat=matrix([[1.,2.1],
        [2. ,1.1],
        [1.3,1. ],
        [1. ,1. ],
        [2. ,1. ]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels