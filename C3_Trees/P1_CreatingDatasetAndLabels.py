# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
#创建一个简单的数据集
#数据集中包含两个特征'no surfacing','flippers';
#数据的类标签有两个'yes','no'
def creatDataSet():
    dataSet=[[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels=['no surfacing','flippers']
    #返回数据集和类标签
    return dataSet,labels
