# coding:utf8
'''
Created on 2018年3月2日
@author: XuXianda

'''
import PCA
#dataMat：数据集矩阵
dataMat=PCA.loadDataSet('testSet.txt', delim='\t')
#lowDMat：降维后的矩阵
#reconMat：反构矩阵
lowDMat,reconMat=PCA.pca(dataMat, 1)
print('dataMat after PCA:')
print lowDMat
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()
