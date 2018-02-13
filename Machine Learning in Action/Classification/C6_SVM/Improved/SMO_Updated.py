# coding:utf8
'''
Created on 2018年2月10日
@author: XuXianda

'''
from numpy import *
import AuxiliaryFunctions
#SMO外循环代码
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    #保存关键数据
    oS=AuxiliaryFunctions.optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True;alphaPairsChanged=0
    #选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        #没有alpha更新对
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print("fullSet,iter: %d i:%d,pairs changed %d" %\
                (iter,i,alphaPairsChanged))
        else:
            #统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("non-bound,iter: %d i:%d,pairs changed %d" %\
                (iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:entireSet=False
        #如果本次循环没有改变的alpha对，将entireSet置为true，
        #下个循环仍遍历数据集
        elif (alphaPairsChanged==0):entireSet=True
        print("iteration number: %d" %iter)
    return oS.b,oS.alphas

#内循环寻找alphaj
def innerL(i,oS):
    #计算误差
    Ei=AuxiliaryFunctions.calcEk(oS,i)
    #违背kkt条件
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej=AuxiliaryFunctions.selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
        #计算上下界
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print("L==H");return 0
        #计算两个alpha值
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-\
            oS.X[j,:]*oS.X[j,:].T
        if eta>=0:print("eta>=0");return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=AuxiliaryFunctions.clipAlpha(oS.alphas[j],H,L)
        AuxiliaryFunctions.updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not moving enough");return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*\
            (alphaJold-oS.alphas[j])
        AuxiliaryFunctions.updateEk(oS,i)
        #在这两个alpha值情况下，计算对应的b值
        #注，非线性可分情况，将所有内积项替换为核函数K[i,j]
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.X[i,:]*oS.X[i,:].T-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.X[i,:]*oS.X[j,:].T-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i])and (oS.C>oS.alphas[i]):oS.b=b1
        elif(0<oS.alphas[j])and (oS.C>oS.alphas[j]):oS.b=b2
        else:oS.b=(b1+b2)/2.0
        #如果有alpha对更新
        return 1
            #否则返回0
    else: return 0
