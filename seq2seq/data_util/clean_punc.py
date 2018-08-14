#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 23:36:42 2018

@author: chengyu
"""
from string import punctuation
import pickle
import os
#%%

## already tokenized

#%%

def strip_punc(token_list,punc_list):
    token = list()
    for i,t in enumerate(token_list):
        if t in punc_list:
            if len(token_list) == 1:
                token.extend(' ')
            elif i != len(token_list) - 1:
                token.extend(' ')
        else:
            token.append(t)
            
    return token

def save_tokenized_data(train_enc,train_dec,PROCESSED_PATH):
    
    train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = [],[],[],[]
    save_file_path = os.path.join(PROCESSED_PATH,'processed_tokens_clean.p')
    
    train_enc_tokens = [strip_punc(t,punc_list) for t in train_enc]
    print('Train_enc_token done.')
    train_dec_tokens = train_dec
    
    pickle.dump((train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens),open(save_file_path,'wb'))
    
    return train_enc_tokens, train_dec, test_enc_tokens,test_dec_tokens

#%%
punc_list = punctuation + "？，。/、「·`！@#￥%……&×（）"
pickle_path = '../data/processed/processed_tokens.p'
PROCESSED_PATH = "../data/processed/"
train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens= pickle.load(open(pickle_path,'rb'))
#%%
train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = save_tokenized_data(train_enc_tokens,train_dec_tokens,PROCESSED_PATH)

#%%
#i = 106009
#print(train_enc_tokens[i])
#print(strip_punc(train_enc_tokens[i],punc_list))