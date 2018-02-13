# coding:utf8
'''
Created on 2018年2月1日
@author: XuXianda

'''
from numpy import *
import P1_LoadDataset
import P2_CreateVocabList
import P3_CreateVectors
import P4_Training
import P5_Classifying
#分类测试整体函数        
def testingNB():
    #由数据集获取文档矩阵和类标签向量
    listOPosts,listClasses=P1_LoadDataset.loadDataSet()
    #统计所有文档中出现的词条，存入词条列表
    myVocabList=P2_CreateVocabList.createVocabList(listOPosts)
    #创建新的列表
    trainMat=[]
    for postinDoc in listOPosts:
        #将每篇文档利用words2Vec函数转为词条向量，存入文档矩阵中
        trainMat.append(P3_CreateVectors.setOfWords2Vec(myVocabList,postinDoc))\
    #将文档矩阵和类标签向量转为NumPy的数组形式，方便接下来的概率计算
    #调用训练函数，得到相应概率值
    p0V,p1V,pAb=P4_Training.trainNB0(array(trainMat),array(listClasses))
    #测试文档
    testEntry=['love','my','dalmation']
    #将测试文档转为词条向量，并转为NumPy数组的形式
    thisDoc=array(P3_CreateVectors.setOfWords2Vec(myVocabList,testEntry))
    #利用贝叶斯分类函数对测试文档进行分类并打印
    print(testEntry,'classified as:',P5_Classifying.classifyNB(thisDoc,p0V,p1V,pAb))
    #第二个测试文档
    testEntry1=['stupid','garbage']
    #同样转为词条向量，并转为NumPy数组的形式
    thisDoc1=array(P3_CreateVectors.setOfWords2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',P5_Classifying.classifyNB(thisDoc1,p0V,p1V,pAb))
#触发
testingNB()
