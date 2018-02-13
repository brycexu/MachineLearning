# coding:utf8
'''
Created on 2018年2月1日
@author: XuXianda

'''
from numpy import *
#训练算法，从词向量计算概率p(w0|ci)...及p(ci)
#@trainMatrix：由每篇文档的词条向量组成的文档矩阵
#@trainCategory:每篇文档的类标签组成的向量
def trainNB0(trainMatrix,trainCategory):
    #获取文档矩阵中文档的数目
    numTrainDocs=len(trainMatrix)
    #获取词条向量的长度
    numWords=len(trainMatrix[0])
    #所有文档中属于类1所占的比例p(c=1)
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #创建一个长度为词条向量等长的列表
    #p0Num=zeros(numWords);p1Num=zeros(numWords)
    #p0Denom=0.0;p1Denom=0.0
    #部分改进1
    p0Num=ones(numWords);p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    #遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        #如果该词条向量对应的标签为1
        if trainCategory[i]==1:
            #统计所有类别为1的词条向量中各个词条出现的次数
            p1Num+=trainMatrix[i]
            #统计类别为1的词条向量中出现的所有词条的总数
            #即统计类1所有文档中出现单词的数目
            p1Denom+=sum(trainMatrix[i])
        else:
            #统计所有类别为0的词条向量中各个词条出现的次数
            p0Num+=trainMatrix[i]
            #统计类别为0的词条向量中出现的所有词条的总数
            #即统计类0所有文档中出现单词的数目
            p0Denom+=sum(trainMatrix[i])
    #利用NumPy数组计算p(wi|c1)
    #部分改进2
    #p1Vect=p1Num/p1Denom  #为避免下溢出问题，后面会改为log()
    p0Vect=log(p0Num/p0Denom);
    #利用NumPy数组计算p(wi|c0)
    #p0Vect=p0Num/p0Denom  #为避免下溢出问题，后面会改为log()
    p1Vect=log(p1Num/p1Denom);
    return p0Vect,p1Vect,pAbusive
