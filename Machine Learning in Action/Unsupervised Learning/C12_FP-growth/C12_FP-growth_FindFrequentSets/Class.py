# coding:utf8
'''
Created on 2018年2月26日
@author: XuXianda

'''
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue    #节点元素名称
        self.count=numOccur    #出现次数
        self.nodeLink=None     #指向下一个节点的指针
        self.parent=parentNode #指向父节点的指针
        self.children={}       #指向子节点的指针
    
    def inc(self,numOccur):
        #增加节点的出现次数
        self.count+=numOccur
        
    def disp(self,ind=1):
        #输出节点和子节点的FP树结构
        print ' ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)
