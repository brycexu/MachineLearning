# coding:utf8
'''
Created on 2018年2月14日
@author: XuXianda

'''
from numpy import *
import P2_ChooseBestSplit
import P3_SplitDataset
#创建树函数
#@dataSet：数据集
#@leafType：生成叶节点的类型 1 回归树：叶节点为常数值 2 模型树：叶节点为线性模型
#@errType：计算误差的类型 1 回归错误类型：总方差=均方差*样本数
#                         2 模型错误类型：预测误差(y-yHat)平方的累加和
#@ops：用户指定的参数，包含tolS：容忍误差的降低程度 tolN：切分的最少样本数
def createTree(dataSet,leafType=P2_ChooseBestSplit.regLeaf,errType=P2_ChooseBestSplit.regErr,ops=(1,4)):
    #选取最佳分割特征和特征值
    feat,val=P2_ChooseBestSplit.chooseBestSplit(dataSet,leafType,errType,ops)
    #如果特征为none，直接返回叶节点值
    if feat == None:return val
    #树的类型是字典类型
    retTree={}
    #树字典的第一个元素是切分的最佳特征
    retTree['spInd']=feat
    #第二个元素是最佳特征对应的最佳切分特征值
    retTree['spval']=val
    #根据特征索引及特征值对数据集进行二元拆分，并返回拆分的两个数据子集
    lSet,rSet=P3_SplitDataset.binSplitDataSet(dataSet,feat,val)
    #第三个元素是树的左分支，通过lSet子集递归生成左子树
    retTree['left']=createTree(lSet,leafType,errType,ops)
    #第四个元素是树的右分支，通过rSet子集递归生成右子树
    retTree['right']=createTree(rSet,leafType,errType,ops)
    #返回生成的数字典
    return retTree