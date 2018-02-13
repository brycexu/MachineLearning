# coding:utf8
'''
Created on 2018年2月12日
@author: XuXianda

'''
import P1_CreatingDatasetAndLabels
import P3_ImplementingAdaBoost
import P4_Testing
#创建数据集合和对应的标签集合
dataArr,labelArr=P1_CreatingDatasetAndLabels.loadSimpData()
#创建对应弱分类器数组
classifierArr=P3_ImplementingAdaBoost.adaBoostTrainDS(dataArr, labelArr, 30)
#进行分类
print("Result:")
print P4_Testing.adaClassify([[5,5],[0,0]], classifierArr)