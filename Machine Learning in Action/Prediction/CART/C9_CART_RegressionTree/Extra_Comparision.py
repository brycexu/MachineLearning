# coding:utf8
'''
Created on 2018年2月17日
@author: XuXianda

'''
from numpy import *
import P1_CreatingDataset
import P4_CreatingRegressionTree
#根据目标数据的存储类型是否为字典型，是返回true，否则返回false
def isTree(obj):
    return (type(obj).__name__=='dict')

#进行树预测
#回归树的叶节点为float型常量
def regTreeEval(model,inDat):
    return float(model)

#树预测    
#@tree；树回归模型
#@inData：输入数据
#@modelEval：叶节点生成类型，需指定，默认回归树类型
def treeForeCast(tree,inData,modelEval=regTreeEval):
    #如果当前树为叶节点，生成叶节点
    if not isTree(tree):return modelEval(tree,inData)
    #非叶节点，对该子树对应的切分点对输入数据进行切分
    if inData[tree['spInd']]>tree['spval']:
        #该树的左分支为非叶节点类型
        if isTree(tree['left']):
            #递归调用treeForeCast函数继续树预测过程，直至找到叶节点
            return treeForeCast(tree['left'],inData,modelEval)
        #左分支为叶节点，生成叶节点
        else: return modelEval(tree['left'],inData)
    #小于切分点值的右分支
    else:
        #非叶节点类型
        if isTree(tree['right']):
            #继续递归treeForeCast函数寻找叶节点
            return treeForeCast(tree['right'],inData,modelEval)
        #叶节点，生成叶节点类型
        else: return modelEval(tree['right'],inData)

#创建预测树        
def createForeCast(tree,testData,modelEval=regTreeEval):
    #测试集样本数
    m=len(testData)
    #初始化行向量各维度值为1
    yHat=mat(zeros((m,1)))
    #遍历每个样本
    for i in range(m):
        #利用树预测函数对测试集进行树构建过程，并计算模型预测值
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    #返回预测值
    return yHat

#树预测检测
myTrainDataset=P1_CreatingDataset.loadDatabase('bikeSpeedVsIq_train.txt')
myTrainDatasetMat=mat(myTrainDataset)
myTestDataset=P1_CreatingDataset.loadDatabase('bikeSpeedVsIq_test.txt')
myTestDatasetMat=mat(myTestDataset)
myRegressionTree=P4_CreatingRegressionTree.createTree(myTrainDatasetMat,ops=(1,20))
yHat=createForeCast(myRegressionTree, myTestDatasetMat[:,0])
print corrcoef(yHat,myTestDatasetMat[:,1],rowvar=0)[0,1]

