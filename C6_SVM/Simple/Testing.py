# coding:utf8
'''
Created on 2018年2月10日
@author: XuXianda

'''
import SMO_Simple
import AuxiliaryFunctions
dataArr,labelArr=AuxiliaryFunctions.loadDataSet("testSet.txt")
b,alphas=SMO_Simple.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print("支持向量：")
for i in range(100):
    if alphas[i]>0.0:print dataArr[i],labelArr[i]
