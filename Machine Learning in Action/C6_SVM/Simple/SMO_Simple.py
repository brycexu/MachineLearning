# coding:utf8
'''
Created on 2018年2月10日
@author: XuXianda

'''
#SMO算法的伪代码
#创建一个alpha向量并将其初始化为0向量
#当迭代次数小于最大迭代次数时(w外循环)
    #对数据集中每个数据向量(内循环):
    #如果该数据向量可以被优化：
        #随机选择另外一个数据向量
        #同时优化这两个向量
        #如果两个向量都不能被优化，退出内循环
#如果所有向量都没有被优化，增加迭代次数，继续下一次循环

#@dataMat    ：数据列表
#@classLabels：标签列表
#@C          ：权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）
#@toler      ：容错率
#@maxIter    ：最大迭代次数
import AuxiliaryFunctions
from numpy import *
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    #将列表形式转为矩阵或向量形式
    dataMatrix=mat(dataMatIn);labelMat=mat(classLabels).transpose()
    #初始化b=0，获取矩阵行列
    b=0;m,n=shape(dataMatrix)
    #新建一个m行1列的向量
    alphas=mat(zeros((m,1)))
    #迭代次数为0
    iter=0
    while(iter<maxIter):
        #改变的alpha对数
        alphaPairsChanged=0
        #遍历样本集中样本
        for i in range(m):
            #计算支持向量机算法的预测值
            fXi=float(multiply(alphas,labelMat).T*\
            (dataMatrix*dataMatrix[i,:].T))+b
            #计算预测值与实际值的误差
            Ei=fXi-float(labelMat[i])
            #如果不满足KKT条件，即labelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            #and alpha<C 或者labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if((labelMat[i]*Ei<-toler)and(alphas[i]<C))or\
            ((labelMat[i]*Ei>toler)and(alphas[i]>0)):
                #随机选择第二个变量alphaj
                j=AuxiliaryFunctions.selectJrand(i,m)
                #计算第二个变量对应数据的预测值
                fXj=float(multiply(alphas,labelMat).T*\
                    (dataMatrix*dataMatrix[j,:].T))+b
                #计算与测试与实际值的差值
                Ej=fXj-float(labelMat[j])
                #记录alphai和alphaj的原始值，便于后续的比较
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                #如果两个alpha对应样本的标签不相同
                if(labelMat[i]!=labelMat[j]):
                    #求出相应的上下边界
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H: print("L==H"); continue
                #根据公式计算未经剪辑的alphaj
                #------------------------------------------
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                #如果eta>=0,跳出本次循环
                if eta>=0:print("eta>=0");continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=AuxiliaryFunctions.clipAlpha(alphas[j],H,L)
                #------------------------------------------    
                #如果改变后的alphaj值变化不大，跳出本次循环    
                if(abs(alphas[j]-alphaJold)<0.00001):print("j not moving\
                enough");continue
                #否则，计算相应的alphai值
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                #再分别计算两个alpha情况下对于的b值
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                #如果0<alphai<C,那么b=b1
                if(0<alphas[i]) and (C>alphas[i]):b=b1
                #否则如果0<alphai<C,那么b=b1
                elif (0<alphas[j]) and (C>alphas[j]):b=b2
                #否则，alphai，alphaj=0或C
                else:b=(b1+b2)/2.0
                #如果走到此步，表面改变了一对alpha值
                alphaPairsChanged+=1
                print("iter: %d i:%d,pairs changed %d" % \
                         (iter,i,alphaPairsChanged))
        #最后判断是否有改变的alpha对，没有就进行下一次迭代
        if(alphaPairsChanged==0):iter+=1
        #否则，迭代次数置0，继续循环
        else:iter=0
        print("iteration number: %d" % iter)
    #返回最后的b值和alpha向量
    return b,alphas
