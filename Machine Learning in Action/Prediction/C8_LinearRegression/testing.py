# coding:utf8
'''
Created on 2018年2月13日
@author: XuXianda

'''
import P1_CreatingDatasetAndLabels
import P2_FiguringOutRegressionWeights
from numpy import *
#准备数据
xArr,yArr=P1_CreatingDatasetAndLabels.loadDataSet('ex0.txt')
#回归系数预测
ws=P2_FiguringOutRegressionWeights.standRegres(xArr, yArr)
xMat=mat(xArr)
yMat=mat(yArr)
#yHat：预测的y
yHat=xMat*ws
#绘制出数据集散点图和最佳拟合直线图
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('The Fucking Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=10,c='blue',marker='x')
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.scatter(xCopy[:,1].flatten().A[0],yHat[:,0].flatten().A[0],linewidths=0.1,s=12,c='red',marker='o')
plt.show()
#计算相关系数
print corrcoef(yHat.T,yMat)