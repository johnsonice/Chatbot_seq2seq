#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:16:55 2017

@author: chengyu
"""
import pickle
import jieba 
USER_DICT = './userdict.txt'
jieba.load_userdict(USER_DICT)
#%%
vocab_path = "../data/weibo_single/jason/vocab.p"
tokens_path = "../data/weibo_single/jason/processed_tokens.p"
pickle_in = open(vocab_path,"rb")
vocab_to_int, int_to_vocab = pickle.load(pickle_in)
#%%
train_enc_tokens, train_dec_tokens,test_enc_tokens, test_dec_tokens = pickle.load(open(tokens_path,"rb"))
#%%

test = "_人名_ 你在干嘛呢"
print(list(jieba.cut(test)))