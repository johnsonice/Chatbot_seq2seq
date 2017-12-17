# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:19:51 2017

@author: Chengyu
"""

import pickle
import os 
from collections import Counter

### combine all processed token data and build vocabulary 

data_path = ['../data/xiaolajiao/processed/processed_tokens.p','../data/xiaohuangji/processed/processed_tokens.p','../data/weibo/processed/processed_tokens.p']
#data_path = ['../data/xiaohuangji/processed/processed_tokens.p']
PROCESSED_PATH = '../data/processed'

#%%

#############################
    ## combine pickles files 
#############################
def load_training_data(train_token_path):
    train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens= pickle.load(open(train_token_path,'rb'))
    return train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens

def combine_pickles(data_path):
    train_enc,train_dec,test_enc,test_dec = [],[],[],[]
    for p in data_path:
        train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = load_training_data(p)
        train_enc.extend(train_enc_tokens)
        train_dec.extend(train_dec_tokens)
        test_enc.extend(test_enc_tokens)
        test_dec.extend(test_dec_tokens)
        print(len(train_enc), len(train_dec))
    
    assert len(train_enc) == len(train_dec)
    assert len(test_enc) == len(test_dec)
    
    save_file_path = os.path.join(PROCESSED_PATH,'processed_tokens.p')
    pickle.dump((train_enc,train_dec,test_enc,test_dec),open(save_file_path,'wb'))
    return train_enc,train_dec,test_enc,test_dec

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

## so idealy, we want to drop those words that does not happend very often 
def build_vocab(pickle_file_path,CODES):
    tokens = pickle.load(open(pickle_file_path,'rb'))
    all_words = []
    for t in tokens:
        all_words.extend(list(_flatten(t)))
    print('Finish flaten tokens')
    counts = Counter(all_words)
    counts = {x : counts[x] for x in counts if counts[x] > 20 }   ## filter out words only appears once
    print('Create counter')
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, len(CODES))}  # enumerate start from len(CODES)
    vocab_to_int = dict(vocab_to_int,**CODES)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
    
    save_file_path = os.path.join(PROCESSED_PATH,'vocab.p')
    pickle.dump((vocab_to_int,int_to_vocab),open(save_file_path,'wb'))
    
    return vocab_to_int,int_to_vocab

#vocab_to_int,int_to_vocab = build_vocab(os.path.join(config.PROCESSED_PATH,'processed_tokens.p'),CODES)
#%%

_ = combine_pickles(data_path)
print('Finish combining datasets')
del _ ## just clear memory
print('clear memory')

#%%
print('building vocabulary')
vocab_to_int,int_to_vocab = build_vocab(os.path.join(PROCESSED_PATH,'processed_tokens.p'),CODES)

#%%
print(len(vocab_to_int))