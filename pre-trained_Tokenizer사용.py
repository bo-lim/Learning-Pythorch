# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:12:02 2021

@author: ant67
"""
'''
import sentencepiece as spm
s = spm.SentencePieceProcessor(model_file='spm.model')
for n in range(5):
    s.encode('New York',out_type=str,enable_sampling=True,alpha=0.1,nbest=-1)
    
'''
# 5-5_model_imdb_BERT.ipynb Code 확인
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence = "My dog is cute. He likes playing"
print(tokenizer.tokenize(sentence))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
print(len(tokenizer.vocab))
print(tokenizer.tokenize(sentence))

sentence = "나는 책상 위에 사과를 먹었다. 알고 보니 그 사과는 Jason 것이었다. 그래서 Jason에게 사과를 했다."
print(tokenizer.tokenize(sentence))

