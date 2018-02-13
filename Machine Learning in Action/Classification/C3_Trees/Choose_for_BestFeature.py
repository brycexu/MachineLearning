# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
import Calculation_of_ShannonEntropy
import Split_of_Dataset
#如何选择最好的划分数据集的特征
#使用某一特征划分数据集，信息增益最大，则选择该特征作为最优特征
def chooseBestFeatureToSplit(dataSet):
    #获取数据集特征的数目(不包含最后一列的类标签)
    numFeatures=len(dataSet[0])-1
    #计算未进行划分的信息熵
    baseEntropy=Calculation_of_ShannonEntropy.calEnt(dataSet)
    #最优信息增益    最优特征
    bestInfoGain=0.0;bestFeature=-1
    #利用每一个特征分别对数据集进行划分，计算信息增益
    for i in range(numFeatures):
        #得到特征i的特征值列表
        featList=[example[i] for example in dataSet]
        #利用set集合的性质--元素的唯一性，得到特征i的取值
        uniqueVals=set(featList)
        #信息增益0.0
        newEntropy=0.0
        #对特征的每一个取值，分别构建相应的分支
        for value in uniqueVals:
            #根据特征i的取值将数据集进行划分为不同的子集
            #利用splitDataSet()获取特征取值Value分支包含的数据集
            subDataSet=Split_of_Dataset.splitDataSet(dataSet,i,value)
            #计算特征取值value对应子集占数据集的比例
            prob=len(subDataSet)/float(len(dataSet))
            #计算占比*当前子集的信息熵,并进行累加得到总的信息熵
            newEntropy+=prob*Calculation_of_ShannonEntropy.calEnt(subDataSet)
        #计算按此特征划分数据集的信息增益
        #公式特征A,数据集D
        #则H(D,A)=H(D)-H(D/A)
        infoGain=baseEntropy-newEntropy
        #比较此增益与当前保存的最大的信息增益
        if (infoGain>bestInfoGain):
            #保存信息增益的最大值
            bestInfoGain=infoGain
            #相应地保存得到此最大增益的特征i
            bestFeature=i
        #返回最优特征
    return bestFeature
