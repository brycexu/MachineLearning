# coding:utf8
'''
Created on 2018年3月7日
@author: XuXianda

'''
from numpy import *
from numpy import linalg as la

def loadExData2():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

#欧式距离相似度计算
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

#相关系数相似度计算
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

#余弦距离相似度计算
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item):
    #对于未评级物品的评分预测
    #dataMat：数据矩阵
    #user：目标用户编号
    #simMeans：相似度计算方法,默认余弦距离
    #item：物品编号
    n = shape(dataMat)[1]
    #需要更新的两个相似度计算相关的值
    simTotal = 0.0; ratSimTotal = 0.0
    #遍历目标用户的物品列
    for j in range(n):
        #如果目标用户对该物品未评分,则跳出本次循环
        userRating = dataMat[user,j]
        if userRating == 0: continue
        #用'logical_and'函数,统计目标列与当前列中在当前行均有评分的数据
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        #如果不存在,则当前列于目标列相似性为0返回
        if len(overLap) == 0: similarity = 0
        #否则,计算这两列中均有评分的行之间的相似度
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        #更新两个变量的值
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    #返回对于目标用户user的未评级物品item的评分预测
    else: 
        print 'so, the presiction at %d is: %f' % (item,ratSimTotal/simTotal)
        return ratSimTotal/simTotal
    
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    #推荐系统主体函数
    #dataMat：数据矩阵
    #user：用户编号
    #N=3：保留的相似度最高的前N个菜肴,默认为3个
    #simMeas：相似度计算方法,默认余弦距离
    #estMethod：评分方法,默认standEst函数
    #从数据矩阵中找出目标用户user所有未评分菜肴的列
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    #如果没有,表明所有菜肴均有评分
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    #遍历每一个未评分的菜肴
    for item in unratedItems:
        #对于每一个未评分菜肴预估评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        #将该菜肴及其预估评分加入数组列表
        itemScores.append((item, estimatedScore))
    #利用sorted函数对列表中的预估评分由高到低排列，返回前N个菜肴
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def svdEst(dataMat, user, simMeas, item):
    #引入SVD的对于未评级物品的评分预测
    #dataMat：数据矩阵
    #user：目标用户编号
    #simMeas：相似度计算方法
    #item：目标菜肴编号
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    #利用SVD进行奇异值分解
    U,Sigma,VT = la.svd(dataMat)
    #保留前四个特征值,并将特征值转化为方阵
    Sig4 = mat(eye(4)*Sigma[:4]) 
    #将数据矩阵进行映射到低维空间
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    #遍历旧数据矩阵的每一列
    for j in range(n):
        #如果目标用户对该物品未评分,则跳出本次循环
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        #否则,按照相似度计算方法,在新数据矩阵上进行评分
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: 
        print 'so, the presiction at %d is: %f' % (item,ratSimTotal/simTotal)
        return ratSimTotal/simTotal


