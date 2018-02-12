# coding:utf8
'''
Created on 2018年2月2日
@author: XuXianda

'''
from numpy import *
import P1_LoadDataset
def plotBestFit(weights):  
    import matplotlib.pyplot as plt    
    dataMat, labelMat = P1_LoadDataset.loadDataSet()  
    dataArr = array(dataMat)  
    n = shape(dataArr)[0]  
    xcord1 = []; ycord1 = []  
    xcord2 = []; ycord2 = []  
    for i in range(n):  
        if int(labelMat[i]) == 1:  
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])  
        else: xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])  
    fig = plt.figure()  
    ax = fig.add_subplot(111)  
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker='s')  
    ax.scatter(xcord2, ycord2, s = 30, c = 'green') 
    #x轴为特征X1，y轴为特征X2，这里y的取值是令0=w0x0+w1x1+w2x2得到的x2，（0是两个分类的分界处） 
    x = arange(-3.0, 3.0, 0.1)  
    y = (0-weights[0]- weights[1]*x)/weights[2]  
    ax.plot(x, y)  
    plt.xlabel('X1');  
    plt.ylabel('X2');  
    plt.show()  
