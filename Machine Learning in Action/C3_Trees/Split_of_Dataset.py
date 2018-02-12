# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
#划分数据集：按照最优特征划分数据集
#@dataSet:待划分的数据集
#@axis:划分数据集的特征
#@value:特征的取值
def splitDataSet(dataSet,axis,value):
    #需要说明的是,python语言传递参数列表时，传递的是列表的引用
    #如果在函数内部对列表对象进行修改，将会导致列表发生变化，为了
    #不修改原始数据集，创建一个新的列表对象进行操作
    retDataSet=[]
    #提取数据集的每一行的特征向量
    for featVec in dataSet:
        #针对axis特征不同的取值，将数据集划分为不同的分支
        #如果该特征的取值为value
        if featVec[axis]==value:
            #将特征向量的0~axis-1列存入列表reducedFeatVec
            reducedFeatVec=featVec[:axis]
            #将特征向量的axis+1~最后一列存入列表reducedFeatVec
            #extend()是将另外一个列表中的元素（以列表中元素为对象）一一添加到当前列表中，构成一个列表
            #比如a=[1,2,3],b=[4,5,6],则a.extend(b)=[1,2,3,4,5,6]
            reducedFeatVec.extend(featVec[axis+1:])
            #简言之，就是将原始数据集去掉当前划分数据的特征列
            #append()是将另外一个列表（以列表为对象）添加到当前列表中
            ##比如a=[1,2,3],b=[4,5,6],则a.extend(b)=[1,2,3,[4,5,6]]
            retDataSet.append(reducedFeatVec)
    return retDataSet
