# coding:utf8
'''
Created on 2018年3月7日
@author: XuXianda

'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
matData=loadExData()
U,Sigma,VT=la.svd(matData)
print Sigma
SigmaNew=mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
matDataNew=U[:,:3]*SigmaNew*VT[:3,:]
print matDataNew
