# coding:utf8
'''
Created on 2018年2月22日
@author: XuXianda

'''
import CreateRules
import Apriori
import AuxiliaryFunctions
dataSet=AuxiliaryFunctions.loadDataSet()
L,supportData=Apriori.apriori(dataSet,minSupport=0.5)
print('L:')
print L
print('supportData:')
print supportData
rules=CreateRules.generateRules(L, supportData, minConf=0.5)
print('rule1:')
print rules
#rules2=CreateRules.generateRules(L,supportData,minConf=0.5)
#print('rule2:')
#print rules2