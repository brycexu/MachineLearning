# coding:utf8
'''
Created on 2018年2月13日
@author: XuXianda

'''
import CreatingDatasetAndLabels
import StageWise
xArr,yArr=CreatingDatasetAndLabels.loadDataSet('abalone.txt')
print StageWise.stageWise(xArr, yArr, 0.01, 200)
