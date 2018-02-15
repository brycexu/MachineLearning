# coding:utf8
'''
Created on 2018年2月14日
@author: XuXianda

'''
#该函数读取一个以tab键位分隔符的文件，然后通过map(float,curLine)方法将每行内容保存为一组浮点数
#前面，目标变量会单独存放其自己的列表中，但这里的数据会存放在一起
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
