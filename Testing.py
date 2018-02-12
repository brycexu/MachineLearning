# coding:utf8
'''
Created on 2018年1月26日
@author: XuXianda

'''
'''
import KNN
group,labels=KNN.createDataSet()
print(group)
print(labels)
print(KNN.classify0([0,0], group, labels, 3))
'''
import Case1
Case1.classifyPerson()