# coding:utf8
'''
Created on 2018年2月2日
@author: XuXianda

'''
import P1_LoadDataset
import P2_UsingGradientRiseMethod
import P3_Analysing
#dataArr是数据集，是列2行100的矩阵，记录了特征X1和X2的数据
#labelMat是数据标签集，是列1行100的矩阵，记录了每个特征向量的类标签（0或1）
dataArr,labelMat=P1_LoadDataset.loadDataSet()
#利用梯度上升法算出权重矩阵weights，0中是X0的权重，1中是X1的权重，2中是X2的权重
#X0始终为1，所以X0的权重可看作是常数项
#和是w0x0+w1x1+w2x2，在0-0.5，看成类0，在0.5-1，看成类1
weights=P2_UsingGradientRiseMethod.gradAscent(dataArr, labelMat)
print(weights)
#在散点图中画出回归直线图像
P3_Analysing.plotBestFit(weights.getA())
