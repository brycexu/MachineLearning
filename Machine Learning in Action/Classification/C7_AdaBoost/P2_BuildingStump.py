# coding:utf8
'''
Created on 2018年2月12日
@author: XuXianda

'''
from numpy import *
#构建单层分类器
#单层分类器是基于最小加权分类错误率的树桩
#伪代码
#将最小错误率minError设为+∞
#对数据集中的每个特征(第一层特征)：
    #对每个步长(第二层特征)：
        #对每个不等号(第三层特征)：
            #建立一颗单层决策树并利用加权数据集对它进行测试
            #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
#返回最佳单层决策树

#单层决策树的阈值过滤函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #对数据集每一列的各个特征进行阈值过滤
    retArray=ones((shape(dataMatrix)[0],1))
    #阈值的模式，将小于某一阈值的特征归类为-1
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    #将大于某一阈值的特征归类为-1
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

        
def buildStump(dataArr,classLabels,D):
#将数据集和标签列表转为矩阵形式
    dataMatrix=mat(dataArr);labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    #步长或区间总数 最优决策树信息 最优单层决策树预测结果
    numSteps=10.0;bestStump={};bestClasEst=mat(zeros((m,1)))
    #最小错误率初始化为+∞
    minError=inf
    #遍历每一列的特征值
    for i in range(n):
        #找出列中特征值的最小值和最大值
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max()
        #求取步长大小或者说区间间隔
        stepSize=(rangeMax-rangeMin)/numSteps
        #遍历各个步长区间
        for j in range(-1,int(numSteps)+1):
            #两种阈值过滤模式
            for inequal in ['lt','gt']:
            #阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                threshVal=(rangeMin+float(j)*stepSize)
            #选定阈值后，调用阈值过滤函数分类预测
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
            #初始化错误向量
                errArr=mat(ones((m,1)))
            #将错误向量中分类正确项置0
                errArr[predictedVals==labelMat]=0
            #计算"加权"的错误率
                weightedError=D.T*errArr
            #打印相关信息，可省略
            #print("split: dim %d, thresh %.2f,thresh inequal:\
            #    %s, the weighted error is %.3f",
            #    %(i,threshVal,inequal,weightedError))
            #如果当前错误率小于当前最小错误率，将当前错误率作为最小错误率
            #存储相关信息
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']='threshVal'
                    bestStump['ineq']=inequal
    #返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    return bestStump,minError,bestClasEst
