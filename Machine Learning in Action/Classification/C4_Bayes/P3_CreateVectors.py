# coding:utf8
'''
Created on 2018年2月1日
@author: XuXianda

'''
#根据词条列表中的词条是否在文档中出现(出现1，未出现0)，将文档转化为词条向量  
#词集模型，即对于一篇文档，将文档中是否出现某一词条作为特征，即特征只能为0不出现或者1出现  
def setOfWords2Vec(vocabSet,inputSet):
    #新建一个长度为vocabSet的列表，并且各维度元素初始化为0
    returnVec=[0]*len(vocabSet)
    #遍历文档中的每一个词条
    for word in inputSet:
        #如果词条在词条列表中出现
        if word in vocabSet:
            #通过列表获取当前word的索引(下标)
            #将词条向量中的对应下标的项由0改为1
            returnVec[vocabSet.index(word)]=1
        else: print('the word: %s is not in my vocabulary! '%'word')
    #返回inputet转化后的词条向量
    return returnVec

#词袋模型，在词袋向量中每个词可以出现多次，这样，在将文档转为向量时，每当遇到一个单词时，它会增加词向量中的对应值
def bagOfWords2VecMN(vocabList,inputSet):
    #词袋向量
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            #某词每出现一次，次数加1
            returnVec[vocabList.index(word)]+=1
    return returnVec