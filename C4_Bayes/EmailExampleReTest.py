# coding:utf8
'''
Created on 2018年2月1日
@author: XuXianda

'''
import re
mySent='This book is the best book on Python or M.L. I have ever laid eyes on.'
print(mySent.split())
regEx=re.compile('\W*')
listOfTokens=regEx.split(mySent)
print(listOfTokens)
