#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:35:15 2017

@author: huang
"""

### helper.py, process data, and batch data 
import os 
import random 
import re 
import numpy as np
import config 
import pickle 
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize


#%%
#########################################
## process cornell movie - dialogs data #
#########################################

## get all sentences with sentence id 
def get_lines():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    with open(file_path, 'r',encoding='utf-8',errors='replace') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line

id2line = get_lines()

#%%

## get all conversion with sentence id in a list
def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos

convos = get_convos()
#%%
def context_answers(convos,id2line):
    context,answers = [],[]
    for convo in convos:
        for index,line in enumerate(convo[:-1]):
            context.append([id2line[line] for line in convo[:index+1]])
            answers.append(id2line[convo[index+1]])
        
    assert len(context) == len(answers)
    return context,answers

context,answers =  context_answers(convos,id2line)

#%%
def _make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
            

def train_test_split(context,answers):
    """
    devide dateset into training and test sets 
    """
    print("Saving to txt pickle")
    # create directory to hold processed data
    _make_dir(config.PROCESSED_PATH)
    
    # random convos to create test set 
    total_numbers = len(context)
    test_size = int(total_numbers * config.TESTSET_SIZE)
    test_ids = random.sample([i for i in range(total_numbers)],test_size)
    
    train_enc, train_dec, test_enc,test_dec = [],[],[],[]
    for i in range(total_numbers):
        if i in test_ids:
            test_enc.append(context[i])
            test_dec.append(answers[i])
        else:
            train_enc.append(context[i])
            train_dec.append(answers[i])
        
        if i % 10000 == 0 : print('Finishing: ',i)
    
    save_file_path = os.path.join(config.PROCESSED_PATH,'processed_text.p')
    pickle.dump((train_enc, train_dec, test_enc,test_dec),open(save_file_path,'wb'))
    
    return train_enc, train_dec, test_enc,test_dec
    
_ = train_test_split(context,answers)
#%%
def _basic_tokenizer(line,normalize_digits=True):
    """
    A basic tokenizer to tokenize text into tokens
    """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    
    _DIGIT_RE = re.compile(r"\d+")  ## find digits 
    
    words = []
    tokens = word_tokenize(line.strip().lower())
    if normalize_digits:
        for token in tokens:
            m = _DIGIT_RE.search(token)
            if m is None:
                words.append(token)
            else:
                words.append('#')
    else:
        words = tokens 
    
    return words 

#test = 'this is the gdp of America 1234%ad. it is going to grow at 5% each year from now on.'
#print(_basic_tokenizer(test))

#%%
def save_tokenized_data(text_pickle_path):
    train_enc, train_dec, test_enc,test_dec = pickle.load(open(text_pickle_path,'rb'))
    train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = [],[],[],[]
    save_file_path = os.path.join(config.PROCESSED_PATH,'processed_tokens.p')
    
    for t in train_enc:
        enc_convo = [_basic_tokenizer(i) for i in t]
        train_enc_tokens.append(enc_convo)
    print('Train_enc_token done.')
    
    for t in train_dec:
        enc_convo = _basic_tokenizer(t)
        train_dec_tokens.append(enc_convo)
    print('Train_dec_token done.')
    
    for t in test_enc:
        enc_convo = [_basic_tokenizer(i) for i in t]
        test_enc_tokens.append(enc_convo)
    print('Test_enc_token done.')
    
    for t in test_dec:
        enc_convo = _basic_tokenizer(t)
        test_dec_tokens.append(enc_convo)
    print('Test_dec_token done.')
    
    pickle.dump((train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens),open(save_file_path,'wb'))
    
    return train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens

## load processed_text and save processed_tokenize
_ = save_tokenized_data(os.path.join(config.PROCESSED_PATH,'processed_text.p'))
    
#%%
########################
## Now build vocabulary 
########################
CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

## a recursive function to flatten nested lists 
def _flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in _flatten(i):
                yield j
        else:
            yield i

def build_vocab(pickle_file_path,CODES):
    tokens = pickle.load(open(pickle_file_path,'rb'))
    all_words = []
    for t in tokens:
        all_words.extend(list(_flatten(t)))
    
    counts = Counter(all_words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, len(CODES))}  # enumerate start from len(CODES)
    vocab_to_int = dict(vocab_to_int,**CODES)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
    
    save_file_path = os.path.join(config.PROCESSED_PATH,'vocab.p')
    pickle.dump((vocab_to_int,int_to_vocab),open(save_file_path,'wb'))
    
    return vocab_to_int,int_to_vocab

vocab_to_int,int_to_vocab = build_vocab(os.path.join(config.PROCESSED_PATH,'processed_tokens.p'),CODES)

#%%
    
def load_vocab(vocab_path):
    vocab_to_int,int_to_vocab = pickle.load(open(vocab_path,'rb'))
    
    
    


            
            
            
            
            
            
            