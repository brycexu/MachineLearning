# coding:utf8
'''
Created on 2018年2月17日
@author: XuXianda

'''
#解析文本数据
def loadDatabase(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        #将每行数据映射为浮点数
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat
