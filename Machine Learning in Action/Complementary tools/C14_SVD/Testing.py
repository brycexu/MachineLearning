# coding:utf8
'''
Created on 2018年3月7日
@author: XuXianda

'''
import SVD
from numpy import *
from numpy import linalg as la
matData=mat(SVD.loadExData2())
#我们试一下默认的推荐
print 'The default recommendation:'
print SVD.recommend(matData,2)
#然后,我们试一下使用SVD的推荐效果
U,Sigma,VT=la.svd(matData)
print 'The recommendation using SVD:'
print SVD.recommend(matData,2,estMethod=SVD.svdEst)
