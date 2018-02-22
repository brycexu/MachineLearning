# coding:utf8
'''
Created on 2018年2月21日
@author: XuXianda

'''
import Apriori
import AuxiliaryFunctions
dataSet=AuxiliaryFunctions.loadDataSet()
print dataSet
C1=AuxiliaryFunctions.createC1(dataSet)
print('C1:')
print C1
D=map(set, dataSet)
print('D:')
print D
L1,supportData1=AuxiliaryFunctions.scanD(D,C1,0.5)
print('L1:')
print L1
print('supportData1:')
print supportData1
L,supportData=Apriori.apriori(dataSet, 0.5)
print('频繁集:')
print('k=1:')
print L[0]
print('k=2:')
print L[1]
print('k=3:')
print L[2]
print('支持度:')
print(supportData)
print('候选集:')
print('C1:')
print C1
print('C2:')
print Apriori.aprioriGen(L[0], 2)
print('C3:')
print Apriori.aprioriGen(L[1], 3)

