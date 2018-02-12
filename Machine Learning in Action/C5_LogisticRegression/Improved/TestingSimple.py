# coding:utf8
'''
Created on 2018年2月2日
@author: XuXianda

'''
import P1_LoadDataset
import P2_UsingGradientRiseMethod
import P3_Analysing
from numpy import *
dataArr,labelMat=P1_LoadDataset.loadDataSet()
weights=P2_UsingGradientRiseMethod.stocGradAscent(dataArr, labelMat)
print(weights)
P3_Analysing.plotBestFit(weights)
weights2=P2_UsingGradientRiseMethod.stocGradAscent1(dataArr, labelMat)
print(weights2)
P3_Analysing.plotBestFit(weights2)