# coding:utf8
'''
Created on 2018年2月13日
@author: XuXianda

'''
import CreatingDatasetAndLabels
import RidgeRegression
abX,abY=CreatingDatasetAndLabels.loadDataSet('abalone.txt')
print RidgeRegression.ridgeRegres(abX, abY, 0.2)
print RidgeRegression.ridgeRegres(abX, abY, 0.1)