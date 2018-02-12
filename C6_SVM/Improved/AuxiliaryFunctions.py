# coding:utf8
'''
Created on 2018年2月10日
@author: XuXianda

'''
from numpy import *
#启发式SMO算法的支持函数
#新建一个类的收据结构，保存当前重要的值
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
#格式化计算误差的函数，方便多次调用
def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*\
        (oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek
#修改选择第二个变量alphaj的方法
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    #将误差矩阵每一行第一列置1，以此确定出误差不为0
    #的样本
    oS.eCache[i]=[1,Ei]
    #获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    #在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if(len(validEcacheList)>0):
        for k in validEcacheList:
            if k ==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
    #否则，就从样本集中随机选取alphaj
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej
#解析文本数据函数，提取每个样本的特征组成向量，添加到数据矩阵
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append((float(lineArr[2])))
    return dataMat,labelMat
#在样本集中采取随机选择的方法选取第二个不等于第一个alphai的优化向量alphaj

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j
#约束范围L<=alphaj<=H内的更新后的alphaj值
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
#更新误差矩阵
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]
