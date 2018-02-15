# coding:utf8
'''
Created on 2018年2月14日
@author: XuXianda

'''

#选取最佳切分特征和最佳特征取值函数

from numpy import *
import P3_SplitDataset
#叶节点生成函数
def regLeaf(dataSet):
    #数据集列表目标变量的均值作为叶节点返回
    return mean(dataSet[:,-1])

#误差计算函数    
def regErr(dataSet):
    #计算数据集目标变量的均值的均方差*数据集样本数，得到总方差返回
    return var(dataSet[:,-1])*shape(dataSet)[0]

#选择最佳切分特征和最佳特征取值函数
#@dataSet：数据集
#@leafType：生成叶节点的类型，默认为回归树类型
#@errType：计算误差的类型，默认为总方差类型
#@ops：用户指定的参数，默认tolS=1.0，tolN=4
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #容忍误差下降值1，最少切分样本数4
    tolS=ops[0];tolN=ops[1]
    #数据集最后一列所有的值都相同
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        #最优特征返回none，将该数据集最后一列计算均值作为叶节点值返回
        return None,leafType(dataSet)
    #数据集的行与列
    m,n=shape(dataSet)
    #计算未切分前数据集的误差
    S=errType(dataSet)
    #初始化最小误差；最佳切分特征索引；最佳切分特征值
    bestS=inf;bestIndex=0;bestValue=0
    #遍历数据集所有的特征，除最后一列目标变量值
    for featIndex in range(n-1):
        #遍历该特征的每一个可能取值
        for splitVal in set(dataSet[:,featIndex]):
            #以该特征，特征值作为参数对数据集进行切分为左右子集
            mat0,mat1=P3_SplitDataset.binSplitDataSet(dataSet,featIndex,splitVal)
            #如果左分支子集样本数小于tolN或者右分支子集样本数小于tolN，跳出本次循环
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):continue
            #计算切分后的误差，即均方差和
            newS=errType(mat0)+errType(mat1)
            #保留最小误差及对应的特征及特征值
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    #如果切分后比切分前误差下降值未达到tolS
    if (S-bestS)<tolS:
        #不需切分，直接返回目标变量均值作为叶节点
        return     None,leafType(dataSet)
    #检查最佳特征及特征值是否满足不切分条件
    mat0,mat1=P3_SplitDataset.binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    #返回最佳切分特征及最佳切分特征取值
    return bestIndex,bestValue

#在选取最佳切分特征和特征值过程中，有三种情况不会对数据集进行切分，而是直接创建叶节点
#（1）如果数据集切分之前，该数据集样本所有的目标变量值相同，那么不需要切分数据集，而直接将目标变量值作为叶节点返回

#（2）当切分数据集后，误差的减小程度不够大（小于tolS）,就不需要切分，而是直接求取数据集目标变量的均值作为叶节点值返回

#（3）当数据集切分后如果某个子集的样本个数小于tolN，也不需要切分，而直接生成叶节点


