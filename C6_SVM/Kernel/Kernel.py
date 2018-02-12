# coding:utf8
'''
Created on 2018年2月12日
@author: XuXianda

'''
from numpy import *
import AuxiliaryFunctions
import SMO_Updated
#径向基核函数是svm常用的核函数
#核转换函数
def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    #如果核函数类型为'lin'
    if kTup[0]=='lin':K=X*A.T
    #如果核函数类型为'rbf':径向基核函数
    #将每个样本向量利用核函数转为高维空间
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:raise NameError('Houston we Have a Problem -- \
    That Kernel is not recognised')
    return K
    
#对核函数处理的样本特征，存入到optStruct中
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def innerL(i,oS):
    #计算误差
    Ei=AuxiliaryFunctions.calcEk(oS,i)
    #违背kkt条件
    if(((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0))):
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
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
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
                    oS.K[i,i]-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.k[i,j]
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.k[i,j]-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.k[i,j]
        if(0<oS.alphas[i])and (oS.C>oS.alphas[i]):oS.b=b1
        elif(0<oS.alphas[j])and (oS.C>oS.alphas[j]):oS.b=b2
        else:oS.b=(b1+b2)/2.0
        #如果有alpha对更新
        return 1
            #否则返回0
    else: return 0

#测试核函数
#用户指定到达率
def testRbf(k1=1.3):
    #第一个测试集
    dataArr,labelArr=AuxiliaryFunctions.loadDataSet('testSetRBF.txt')
    b,alphas=SMO_Updated.smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("there are %d Support Vectors" %shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount+=1
    print("the training error rate is: %f" %(float(errorCount)/m))
    #第二个测试集
    dataArr,labelArr=AuxiliaryFunctions.loadDataSet('testSetRBF2.txt')
    dataMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    errorCount=0
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount+=1
    print("the training error rate is: %f" %(float(errorCount)/m))