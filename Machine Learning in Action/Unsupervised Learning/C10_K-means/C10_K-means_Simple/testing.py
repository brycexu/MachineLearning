# coding:utf8
'''
Created on 2018年2月19日
@author: XuXianda

'''
import KMeans
import AuxiliaryFunctions
from numpy import *
datset=AuxiliaryFunctions.loadDataSet('testSet.txt')
datMat=mat(datset)
myCentroids,clusterAssing=KMeans.kMeans(datMat,4)
print clusterAssing
print myCentroids
n=shape(myCentroids)[0]
point_x=[]
point_y=[]
for i in range(n):
    point_x.append(myCentroids[i,0])
    point_y.append(myCentroids[i,1])
m=shape(clusterAssing)[0]
print m
type1_x=[]
type1_y=[]
type2_x=[]
type2_y=[]
type3_x=[]
type3_y=[]
type4_x=[]
type4_y=[]
for i in range(m):
    if clusterAssing[i,0]==0:
        type1_x.append(datset[i][0])
        type1_y.append(datset[i][1])
    if clusterAssing[i,0]==1:
        type2_x.append(datset[i][0])
        type2_y.append(datset[i][1])
    if clusterAssing[i,0]==2:
        type3_x.append(datset[i][0])
        type3_y.append(datset[i][1])
    if clusterAssing[i,0]==3:
        type4_x.append(datset[i][0])
        type4_y.append(datset[i][1])
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(type1_x,type1_y,c='red')
ax.scatter(type2_x,type2_y,c='green')
ax.scatter(type3_x,type3_y,c='blue')
ax.scatter(type4_x,type4_y,c='orange')
ax.scatter(point_x,point_y,c='black',s=300,marker='x')
ax.set_title('The K-means Method For Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()