# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 10:53:22 2021

@author: ant67
"""

# 1. 단어 횟수를 기록한 사전을 만든다.(띄어쓰기 기반)
# 2. 각 단어에 대해 연속된 2개의 글자의 숫자를 세고 가장 많이 나오는 글자 2개의 조합을 찾는다.(bi-gram)
# 3. 두 글자를 합쳐 기존 사전의 단어를 수정한다.
# 4. 미리 정해 놓은 횟수만큼 2~3번의 과정을 반복한다.

# Algorithm 1: Learn BPE opertaions
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]]+=freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p=re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair),word)
        v_out[w_out]=v_in[word]
    return v_out
'''
vocab = {'l o w </w>': 5, 'l o w e r </w>':2,'n e w e s t </w>':6, 'w i d e s t </w>':3} # 1번 과정
num_merges = 10
for i in range(num_merges): # 4번 과정
    pairs = get_stats(vocab) # 2번 과정
    best = max(pairs, key=pairs.get) # 2번 과정
    vocab = merge_vocab(best,vocab) # 3번 과정
    print(f'Step {i+1}')
    print(best)
    print(vocab)
    print('\\n')
'''    
###################################
S1 = "나는 책상 위에 사과를 먹었다"
S2 = "알고 보니 그 사과는 Jason 것이었다"
S3 = "그래서 Jason에게 사과를 했다."

token_counts = {}
index = 0

for sentence in [S1,S2,S3]:
    tokens = sentence.split()
    for token in tokens:
        if token_counts.get(token) == None:
            token_counts[token] = 1
        else:
            token_counts[token] +=1
print(token_counts)
token_counts = {" ".join(token) : counts for token, counts in token_counts.items()}
print(token_counts)

num_merges=10

for i in range(num_merges):
    pairs = get_stats(token_counts)
    best=max(pairs,key=pairs.get)
    token_counts = merge_vocab(best, token_counts)
    print(f'Step {i+1}')
    print(best)
    print(token_counts)
    print('\\n')















