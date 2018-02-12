# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
import Split_of_Dataset
import Vote_for_Majority
import Choose_for_BestFeature
#创建树
def createTree(dataSet,labels):
    #获取数据集中的最后一列的类标签，存入classList列表
    classList=[example[-1] for example in dataSet]
    #通过count()函数获取类标签列表中第一个类标签的数目
    #判断数目是否等于列表长度，相同表明所有类标签相同，属于同一类
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #遍历完所有的特征属性，此时数据集的列为1，即只有类标签列
    if len(dataSet[0])==1:
        #多数表决原则，确定类标签
        return Vote_for_Majority.majorityCnt(classList)
    #确定出当前最优的分类特征
    bestFeat=Choose_for_BestFeature.chooseBestFeatureToSplit(dataSet)
    #在特征标签列表中获取该特征对应的值
    bestFeatLabel=labels[bestFeat]
    #采用字典嵌套字典的方式，存储分类树信息
    myTree={bestFeatLabel:{}}
    
    ##此位置书上写的有误，书上为del(labels[bestFeat])
    ##相当于操作原始列表内容，导致原始列表内容发生改变
    ##按此运行程序，报错'no surfacing'is not in list
    ##以下代码已改正
    
    #复制当前特征标签列表，防止改变原始列表的内容
    subLabels=labels[:]
    #删除属性列表中当前分类数据集特征
    del(subLabels[bestFeat])
    #获取数据集中最优特征所在列
    featValues=[example[bestFeat] for example in dataSet]
    #采用set集合性质，获取特征的所有的唯一取值
    uniqueVals=set(featValues)
    #遍历每一个特征取值
    for value in uniqueVals:
        #采用递归的方法利用该特征对数据集进行分类
        #@bestFeatLabel 分类特征的特征标签值
        #@dataSet 要分类的数据集
        #@bestFeat 分类特征的标称值
        #@value 标称型特征的取值
        #@subLabels 去除分类特征后的子特征标签列表
        myTree[bestFeatLabel][value]=createTree(Split_of_Dataset.splitDataSet\
            (dataSet,bestFeat,value),subLabels)
    return myTree

#决策树的存储：python的pickle模块序列化决策树对象，使决策树保存在磁盘中
#在需要时读取即可，数据集很大时，可以节省构造树的时间
#pickle模块存储决策树
def storeTree(inputTree,filename):
    #导入pickle模块
    import pickle
    #创建一个可以'写'的文本文件
    #这里，如果按树中写的'w',将会报错write() argument must be str,not bytes
    #所以这里改为二进制写入'wb'
    fw=open(filename,'wb')
    #pickle的dump函数将决策树写入文件中
    pickle.dump(inputTree,fw)
    #写完成后关闭文件
    fw.close()
#取决策树操作    
def grabTree(filename):
    import pickle
    #对应于二进制方式写入数据，'rb'采用二进制形式读出数据
    fr=open(filename,'rb')
    return pickle.load(fr)
