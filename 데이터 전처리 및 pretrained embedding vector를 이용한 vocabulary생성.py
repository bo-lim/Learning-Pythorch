# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:57:03 2021

@author: ant67
"""
import torch
from torchtext import data
from torchtext import datasets

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# Data setting
TEXT = data.Field(batch_first = True,
                  fix_length=500,
                  tokenize=str.split,
                  pad_first=True,
                  pad_token='[PAD]',
                  unk_token='[UNK]')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(text_field=TEXT,
                                             label_field=LABEL)
# Data Length
print(f'Train Data Length : {len(train_data.examples)}')
print(f'Test Data Length : {len(test_data.examples)}')

# Train Data Length: 25000
# Test Data Length: 25000

# Data Fields
print(train_data.fields)

# Data Sample
print('----Data Sample----')
print('Input: ')
print(' '.join(vars(train_data.examples[1])['text']),'\n')
print('Label: ')
print(vars(train_data.examples[1])['label'])


# Cleansing 작업
# Field의 preprocessing 옵션을 이용해도 가능함
import re
def PreProcessingText(input_sentence):
    input_sentence=input_sentence.lower() # 소문자화
    input_sentence = re.sub('<[^>]*>',repl=' ',string = input_sentence) # "<br />처리
    input_sentence = re.sub('[!"#$%&\\()*+,-./:;<=>?@[\\\\]^_{|}~]',repl=' ',string = input_sentence) # 특수문자처리("'"제외)
    input_sentence = re.sub('\\s+',repl=' ',string = input_sentence) # 연속된 띄어쓰기 처리
    if input_sentence:
        return input_sentence
    
for example in train_data.examples:
    vars(example)['text'] = PreProcessingText(' '.join(vars(example)['text'])).split()

for example in test_data.examples:
    vars(example)['text'] = PreProcessingText(' '.join(vars(example)['text'])).split()


# pre-trained
TEXT.build_vocab(train_data,
                 min_freq = 2, # Vocab에 해당하는 Token에 최소한으로 등장하는 회수 제한
                 max_size = None, # 전체 Vocab size 자체에 제한
                 vectors = "glove.6B.300d") # pre=trained Vector를 가져와 Vocab에 세팅하는 옵션. 원한느 Embedding을 string으로 지정

LABEL.build_vocab(train_data)

# voab에 대한 정보 확인

# Vocab Info
print(f'Vocab Size : {len(TEXT.vocab)}')

print('Vocab Examples: ')
for idx,(k,v) in enumerate(TEXT.vocab.stoi.items()):
    if idx >= 10:
        break
    print('\t',k,v)
    
print('--------------------------------------------')

# Label Info
print(f'Label Size: {len(LABEL.vocab)}')

print('Label Examples: ')
for idx,(k,v) in enumerate(LABEL.vocab.stoi.items()):
    print('\t',k,v)
    
# Check embedding vectors
print(TEXT.vocab.vectors.shape)




# Validation set 구분, Iterator를 이용해 Batch Data 만들기
import random

# Spliting Valid set
train_data, valid_data = train_data.split(random_state = random.seed(0),
                                          split_ratio=0.8)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(datasets=(train_data, valid_data,test_data),batch_size = 30, device = device)

class SentenceClassification(nn.Module):
    def __init__(self, **model_config):
        super(SentenceClassification,self).__init__()
        
        if model_config['emb_type'] == 'glove' or 'fasttext':
            self.emb = nn.Embedding(model_config['vocab_size'],
                                    model_config['emb_dim'],
                                    _weight = TEXT.voacb.vectors)
        else:
            self.emb = nn.Embedding(model_config['vocab_size'],
                                    model_config['emb_dim'])
            
        self.bidirectional = model_config['bidirectional']
        self.num_direction = 2 if model_config['bidirectional'] else 1
        self.model_type = model_config['model_type']
        
        self.RNN = nn.RNN(input_size = model_config['emb_dim'],
                          hidden_size = model_config['hidden_dim'],
                          dropout = model_config['dropout'],
                          bidirectional = model_config['bidirectional'],
                          batch_first = model_config['batch_first'])
        
        self.LSTM = nn.LSTM(input_size = model_config['emb_dim'],
                          hidden_size = model_config['hidden_dim'],
                          dropout = model_config['dropout'],
                          bidirectional = model_config['bidirectional'],
                          batch_first = model_config['batch_first'])
        
        self.GRU = nn.GRU(input_size = model_config['emb_dim'],
                          hidden_size = model_config['hidden_dim'],
                          dropout = model_config['dropout'],
                          bidirectional = model_config['bidirectional'],
                          batch_first = model_config['batch_first'])
        
        self.fc = nn.Linear(model_config['hidden_dim'] * self.num_direction,
                            model_config['output_dim'])
        
        self.drop = nn.Dropout(model_config['dropout'])
    def forward(self, x):
        emb=self.emb(x)
        
        if self.model_type == 'RNN':
            output, hidden = self.RNN(emb)
        elif self.model_type == 'LSTM':
            output, hidden = self.LSTM(emb)
        elif self.model_type == 'GRU':
            output, hidden = self.GRU(emb)
        else:
            raise NameError('Select model_type in [RNN, LSTM, GRU]')
            
        last_output = output[:,-1,:]
        
        return self.fc(self.drop(last_output))
    
    
sample_for_check = next(iter(train_iterator))
print(sample_for_check)
print(sample_for_check.text)
print(sample_for_check.label)

model_config=dict(batch_first=True,
                         model_type='RNN',
                         bidirectional = True,
                         hidden_dim=128,
                         output_dim=1,
                         dropout = 0)

model = SentenceClassification(**model_config).to(device)
loss_fn = nn.BCEWithLogitsLoss().to(device)

def binary_accuracy(preds,y):
    rounded_pred = torch.round(torch.sigmoid(preds))
    correct = (rounded_pred == y).float()
    acc= correct.sum()/len(correct)
    return acc

predictions = model.forward(sample_for_check.text).squeeze()
loss = loss_fn(predictions, sample_for_check.label)
acc = binary_accuracy(predictions, sample_for_check.label)

print(predictions)
print(loss,acc)