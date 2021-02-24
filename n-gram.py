# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 01:46:48 2021

@author: ant67
"""
S1 = "나는 책상 위에 사과를 먹었다."

#uni-gram
print([S1[i:i+1] for i in range(len(S1))])
# bi-gram
print([S1[i:i+1] for i in range(len(S1))])
# tri-gram
print([S1[i:i+3] for i in range(len(S1))])
