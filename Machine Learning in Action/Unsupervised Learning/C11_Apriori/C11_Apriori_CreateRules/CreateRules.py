# coding:utf8
'''
Created on 2018年2月22日
@author: XuXianda

'''
import Apriori
def generateRules(L, supportData, minConf=0.7):  
    #生成包含可信度的规则列表
    #L:频繁项集合列表
    #supportData:包含频繁项集合以及对应支持度的字典
    #minConf:最小可信度
    #bigRuleList:盛放关联规则
    bigRuleList = []
    #因为是关联，所以只获取两个或更多元素的集合
    for i in range(1, len(L)):
        for freqSet in L[i]:
            #H1:freqSet中所有元素的单元素组成的集合
            #例:[2,3],H1=[frozenset([2]),frozenset([3])]
            H1 = [frozenset([item]) for item in freqSet]
            print('H1:')
            print H1
            if (i > 1):
                #如果频繁项集合的元素数目超过2,那么考虑对它做进一步的合并,合并通过rulesFromConseq完成
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #如果频繁项集合中只有两个元素，那么直接用calcConf()来计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #计算规则的可信度以及找到满足要求(可信度大于最小可信度)的规则
    prunedH = [] 
    #遍历H中的所有项集并计算它们的可信度值
    for conseq in H:
        #计算公式:
        #P->H:support(P|H)/support(P)
        #比方说freqSet:[2,3],H:[(2),(3)],conseq:(2)
        #conf(3->2)=support[[2,3]]/support[[3]]
        conf = supportData[freqSet]/supportData[freqSet-conseq] 
        #如果满足要求，就加入brl(bigRuleList):最后的输出
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    #输出conseq集合(被指向项)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #生成候选规则集合，从最初的项集中生成更多的关联规则
    #freqSet:频繁项集,例:[2,3,5]
    #H:可以出现在规则右部的元素列表,例:[set([2]),set([3]),set([5])]
    #m:H中频繁项集大小,例:1
    m = len(H[0])
    print('m:')
    print m
    print('freqSet:')
    print freqSet
    #查看freqSet是否可以移除大小为m的子集
    if (len(freqSet) > (m + 1)): 
        #生成H中无重复的m+1元组合,例:[set([2,3]),set([2,5]),set([3,5])]
        Hmp1 = Apriori.aprioriGen(H, m+1)
        print('Hmp1:')
        print Hmp1
        #检测Hmp1中的组合是否能成为规则中的右部(满足最小可信度要求)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        #如果不止一条规则满足要求，那么使用Hmp1迭代
        if (len(Hmp1) > 1):    
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
