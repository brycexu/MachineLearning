# coding:utf8
'''
Created on 2018年1月27日
@author: XuXianda

'''
from C2_KNN import KNN
from numpy import *
from os import listdir
#-------------------------knn算法实例-----------------------------------
#-------------------------手写识别系统-----------------------------------
#-------------------------1 将图像转化为测试向量-------------------------
#图像大小32*32，转化为1024的向量
def img2vector(filename):
    returnVec=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        #每次读取一行
        lineStr=fr.readline()
        for j in range(32):
            #通俗讲：就是根据首地址(位置)的偏移量计算出当前数据存放的地址(位置)
            returnVec[0,32*i+j]=int(lineStr[j])
    return returnVec
#-------------------------2 测试代码--------------------------------------
def handwritingClassTest():
    hwLabels=[]
    #列出给定目录的文件名列表，使用前需导入from os import listdir
    trainingFileList=listdir('knn/trainingDigits')
    #获取列表的长度
    m=len(trainingFileList)
    #创建一个m*1024的矩阵用于存储训练数据
    trainingMat=zeros((m,1024))
    for i in range(m):
        #获取当前行的字符串
        fileNameStr=trainingFileList[i]
        #将字符串按照'.'分开，并将前一部分放于fileStr
        fileStr=fileNameStr.split('.')[0]
        #将fileStr按照'_'分开，并将前一部分存于classNumStr
        classNumStr=int(fileStr.split('_')[0])
        #将每个标签值全部存入一个列表中
        hwLabels.append(classNumStr)
        #解析目录中的每一个文件，将图像转化为向量，最后存入训练矩阵中
        trainingMat[i,:]=img2vector('knn/trainingDigits/%s' %fileNameStr)
    #读取测试数据目录中的文件列表
    testFileList=listdir('knn/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        #获取第i行的文件名
        fileNameStr=testFileList[i]
        #将字符串按照'.'分开，并将前一部分放于fileStr
        fileStr=fileNameStr.split('.')[0]
        #将fileStr按照'_'分开，并将前一部分存于classNumStr
        classNumStr=int(fileStr.split('_')[0])
        #解析目录中的每一个文件，将图像转化为向量
        vectorUnderTest=img2vector('knn/testDigits/%s' %fileNameStr)
        #分类预测
        classifierResult=KNN.classify0(vectorUnderTest,trainingMat,hwLabels,3)
        #打印预测结果和实际结果
        print("the classifierResult came back with: %d,the real answer is: %d" %(classifierResult,classNumStr))
        #预测错误，错误数加1次
        if(classifierResult!=classNumStr):errorCount+=1.0
        #打印错误数和错误率
        print("\nthe total number of errors is: %d" %errorCount)
        print("\nthe total error rate is: %f" %(errorCount/float(mTest)))