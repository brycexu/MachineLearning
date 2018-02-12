# coding:utf8
'''
Created on 2018年2月2日
@author: XuXianda

'''
#预处理数据
def loadDataSet():
    #创建两个列表
    dataMat=[];labelMat=[]
    #打开文本数据集
    fr=open('testSet.txt')
    #遍历文本的每一行
    for line in fr.readlines():
        #对当前行去除首尾空格，并按空格进行分离
        lineArr=line.strip().split()
        #将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        #将当前行标签添加到标签列表
        labelMat.append(int(lineArr[2]))
    #返回数据列表，标签列表
    return dataMat,labelMat
