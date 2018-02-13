# coding:utf8
'''
Created on 2018年1月30日
@author: XuXianda

'''
import operator
#当遍历完所有的特征属性后，类标签仍然不唯一(分支下仍有不同分类的实例)
#采用多数表决的方法完成分类
def majorityCnt(classList):
    #创建一个类标签的字典
    classCount={}
    #遍历类标签列表中每一个元素
    for vote in classList:
        #如果元素不在字典中
        if vote not in classCount.keys():
            #在字典中添加新的键值对
            classCount[vote]=0
        #否则，当前键对于的值加1
        classCount[vote]+=1
    #对字典中的键对应的值所在的列，按照又大到小进行排序
    #@classCount.items 列表对象
    #@key=operator.itemgetter(1) 获取列表对象的第一个域的值
    #@reverse=true 降序排序，默认是升序排序
    sortedClassCount=sorted(classCount.items,\
    key=operator.itemgetter(1),reverse=True)
    #返回出现次数最多的类标签
    return sortedClassCount[0][0]
