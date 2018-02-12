# coding:utf8
'''
Created on 2018年1月27日
@author: XuXianda

'''
from C2_KNN import KNN
from numpy import *
#-------------------------knn算法实例-----------------------------------
#-------------------------约会网站配对-----------------------------------
#---------------1 将text文本数据转化为分类器可以接受的格式---------------
def file2matrix(filename):
    #打开文件
    fr=open(filename)
    #读取文件每一行到array0Lines列表
    #read():读取整个文件，通常将文件内容放到一个字符串中
    #readline():每次读取文件一行，当没有足够内存一次读取整个文件内容时，使用该方法
    #readlines():读取文件的每一行，组成一个字符串列表，内存足够时使用
    array0Lines=fr.readlines()
    #获取字符串列表行数行数
    numberOfLines=len(array0Lines)
    #返回的特征矩阵大小
    returnMat=zeros((numberOfLines,3))
    #list存储类标签
    classLabelVector=[]
    index=0
    for line in array0Lines:
        #去掉字符串头尾的空格，类似于Java的trim()
        line=line.strip()
        #将整行元素按照tab分割成一个元素列表
        listFromLine=line.split('\t')
        #将listFromLine的前三个元素依次存入returnmat的index行的三列
        returnMat[index,:]=listFromLine[0:3]
        #python可以使用负索引-1表示列表的最后一列元素，从而将标签存入标签向量中
        #使用append函数每次循环在list尾部添加一个标签值
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector
#----------------2 准备数据：归一化----------------------------------------------
#计算欧式距离时，如果某一特征数值相对于其他特征数值较大，那么该特征对于结果影响要
#远大于其他特征，然后假设特征都是同等重要，即等权重的，那么可能某一特征对于结果存
#在严重影响
def autoNorm(dataSet):
    #找出每一列的最小值
    minVals=dataSet.min(0)
    #找出每一列的最大值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    #创建与dataSet等大小的归一化矩阵
    #shape()获取矩阵的大小
    normDataSet=zeros(shape(dataSet))
    #获取dataSet第一维度的大小
    m=dataSet.shape[0]
    #将dataSet的每一行的对应列减去minVals中对应列的最小值
    normDataSet=dataSet-tile(minVals,(m,1))
    #归一化，公式newValue=(value-minvalue)/(maxVal-minVal)
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
#-------------------------3 测试算法----------------------------------------------
#改变测试样本占比，k值等都会对最后的错误率产生影响
def datingClassTest():
    #设定用来测试的样本占比
    hoRatio=0.10
    #从文本中提取得到数据特征，及对应的标签
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    #对数据特征进行归一化
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #得到第一维度的大小
    m=normMat.shape[0]
    #测试样本数量
    numTestVecs=int(hoRatio*m)
    #错误数初始化
    errorCount=0.0
    for i in range(numTestVecs):
        #利用分类函数classify0获取测试样本数据分类结果
        classifierResult=KNN.classify0(normMat[i,:],normMat[numTestVecs:m,:],\
        datingLabels[numTestVecs:m],3)
        #打印预测结果和实际标签
        print("the classifier came back with: %d, the real answer is: %d"\
        %(classifierResult,datingLabels[i]))
        #如果预测输出不等于实际标签,错误数增加1.0
        if(classifierResult != datingLabels[i]):errorCount+=1.0
    #打印最后的误差率
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))

#-------------------------4 构建可手动输入系统------------------------------------
#用户输入相关数据，进行预测
def classifyPerson():
    #定义预测结果
    resultList=['not at all','in small does','in large does']
    #在python3.x中，已经删除raw_input()，取而代之的是input()
    percentTats=float(input(\
    "percentage of time spent playing video games?"))
    ffMiles=float(input("frequent filer miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minValues=autoNorm(datingDataMat)
    #将输入的数值放在数组中
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=KNN.classify0((inArr-minValues)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classifierResult-1])
