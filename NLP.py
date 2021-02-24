# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:25:15 2021

@author: ant67
"""
from torchtext import data
from torchtext import datasets

# Data setting
#batch_first : Batch size 를 Data shape Axis의 가장 앞으로 설정하는 옵션
TEXT = data.Field(lower=True, batch_first=True)
LABEL = data.Field(sequential=False)

train, test = datasets.IMDB.splits(TEXT, LABEL)

idx2char = {0:'<pad>', 1:'<unk>'}

srt_idx = len(idx2char)
for x in range(32,127):
    idx2char.update({srt_idx:chr(x)})
    srt_idx +=1

# 한글 추가는 밑의 코드를 실행합니다.
for x in range(int('0x3131',16),int('0x3163',16)+1):
    idx2char.update({srt_idx:chr(x)})
    srt_idx +=1
    
for x in range(int('0xAC00',16),int('0xD7A3',16)+1):
    idx2char.update({srt_idx:chr(x)})
    srt_idx +=1
    
char2idx = {v:k for k,v in idx2char.items()}
print([char2idx.get(c,0) for c in '그래서 Jason에게 사과를 했다'])
print([char2idx.get(c,0) for c in 'ㅇㅋ! ㄳㄳ'])
