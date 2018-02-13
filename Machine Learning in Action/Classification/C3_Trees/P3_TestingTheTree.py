# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
#------------------------测试算法------------------------------    
#完成决策树的构造后，采用决策树实现具体应用
#@intputTree 构建好的决策树
#@featLabels 特征标签列表
#@testVec 测试实例
def classify(inputTree,featLabels,testVec):
    #找到树的第一个分类特征，或者说根节点'no surfacing'
    #注意python2.x和3.x区别，2.x可写成firstStr=inputTree.keys()[0]
    #而不支持3.x
    firstStr=list(inputTree.keys())[0]
    #从树中得到该分类特征的分支，有0和1
    secondDict=inputTree[firstStr]
    #根据分类特征的索引找到对应的标称型数据值
    #'no surfacing'对应的索引为0
    featIndex=featLabels.index(firstStr)
    #遍历分类特征所有的取值
    for key in secondDict.keys():
        #测试实例的第0个特征取值等于第key个子节点
        if testVec[featIndex]==key:
            #type()函数判断该子节点是否为字典类型
            if type(secondDict[key]).__name__=='dict':
                #子节点为字典类型，则从该分支树开始继续遍历分类
                classLabel=classify(secondDict[key],featLabels,testVec)
            #如果是叶子节点，则返回节点取值
            else: classLabel=secondDict[key]
    return classLabel
