# coding:utf8
'''
Created on 2018年1月27日
@author: XuXianda

'''
import matplotlib
import matplotlib.pyplot as plt
import Case1
from numpy import array
datingDataMat,datingLabels=Case1.file2matrix('datingTestSet2.txt')
fig=plt.figure()
ax=fig.add_subplot(111)
#一般绘图：ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#高级绘图：
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
