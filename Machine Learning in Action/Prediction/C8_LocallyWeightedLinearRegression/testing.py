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
#对单点进行估计
print yArr[0]
print P2_FiguringOutRegressionWeights.lwlr(xArr[0], xArr, yArr, 1.0)
print P2_FiguringOutRegressionWeights.lwlr(xArr[0],xArr,yArr,0.001)
#对数据集进行估计
yHat=P2_FiguringOutRegressionWeights.lwlrTest(xArr, xArr, yArr,0.003)
#绘制出图形
#先需要将数据点按序排列
xMat=mat(xArr)
srtInd=xMat[:,1].argsort(0)
xSort=xMat[srtInd][:,0,:]
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('The Fucking LWLR At 0.003')
plt.xlabel('X')
plt.ylabel('Y')
#flatten()是对矩阵进行降维打击到1维
ax.scatter(xSort[:,1].flatten().A[0],yHat[srtInd][:,0],s=12,linewidths=0.1,c='red')
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=8,c='blue',marker='x')
plt.show()
#计算相关系数
print corrcoef(yHat.T,mat(yArr).T.flatten())
