# coding:utf8
'''
Created on 2018年2月20日
@author: XuXianda

'''
import AuxiliaryFunctions
import biKmeans
from numpy import *
datset=AuxiliaryFunctions.loadDataSet('testSet2.txt')
datMat=mat(datset)
centList,myNewAssments=biKmeans.biKmeans(datMat,3)
print centList
print myNewAssments
n=shape(centList)[0]
point_x=[]
point_y=[]
for i in range(n):
    point_x.append(centList[i,0])
    point_y.append(centList[i,1])
m=shape(myNewAssments)[0]
print point_x
print n
print m
type1_x=[]
type1_y=[]
type2_x=[]
type2_y=[]
type3_x=[]
type3_y=[]
for i in range(m):
    if myNewAssments[i,0]==0:
        type1_x.append(datset[i][0])
        type1_y.append(datset[i][1])
    if myNewAssments[i,0]==1:
        type2_x.append(datset[i][0])
        type2_y.append(datset[i][1])
    if myNewAssments[i,0]==2:
        type3_x.append(datset[i][0])
        type3_y.append(datset[i][1])
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(type1_x,type1_y,c='red')
ax.scatter(type2_x,type2_y,c='green')
ax.scatter(type3_x,type3_y,c='blue')
ax.scatter(point_x,point_y,c='black',s=300,marker='x')
ax.set_title('The biK-means Method For Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
