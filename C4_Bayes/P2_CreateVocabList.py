# coding:utf8
'''
Created on 2018年2月1日
@author: XuXianda

'''
#统计所有文档中出现的词条列表    
def createVocabList(dataSet):
    #新建一个存放词条的集合
    vocabSet=set([])
    #遍历文档集合中的每一篇文档
    for document in dataSet:
        #将文档列表转为集合的形式，保证每个词条的唯一性
        #然后与vocabSet取并集，向vocabSet中添加没有出现
        #的新的词条        
        vocabSet=vocabSet|set(document)
    #再将集合转化为列表，便于接下来的处理
    return list(vocabSet)
