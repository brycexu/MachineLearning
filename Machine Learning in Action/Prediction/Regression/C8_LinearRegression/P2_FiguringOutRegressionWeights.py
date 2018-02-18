# coding:utf8
'''
Created on 2018年2月13日
@author: XuXianda

'''
from numpy import *
#标准线性回归算法
#ws=(X.T*X).I*(X.T*Y)    
def standRegres(xArr,yArr):
    #将列表形式的数据转为numpy矩阵形式
    xMat=mat(xArr);yMat=mat(yArr).T
    #求矩阵的内积
    xTx=xMat.T*xMat
    #numpy线性代数库linalg
    #调用linalg.det()计算矩阵行列式
    #计算矩阵行列式是否为0
    if linalg.det(xTx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return 
    #如果可逆，根据公式计算回归系数
    ws=xTx.I*(xMat.T*yMat)
    #可以用yHat=xMat*ws计算实际值y的预测值
    #返回归系数
    return ws
