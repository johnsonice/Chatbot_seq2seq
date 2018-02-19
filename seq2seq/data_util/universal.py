# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:45:30 2018

@author: chuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:35:15 2017

@author: huang
"""

import os 
#####################################
## this is based on the file structure 
# os.chdir('../')
#####################################
import re 
import pickle 
import csv
import jieba
USER_DICT = 'userdict.txt'
jieba.load_userdict(USER_DICT)


DELETE = ['\ufeff']

#%%
def get_lines(data_path):
    with open(data_path,'r',encoding='utf-8',errors='replace') as f:
        reader = list(csv.reader(f))
        context = [r[0].replace('\ufeff','') for r in reader]
        answers = [r[1] for r in reader]
        
    assert len(context) == len(answers)
    return context,answers

#%%
def _make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
            
#%%

## for xiaohuangji data, it has already been tokenized 
## do do not need to run tokenizer
def _basic_tokenizer(line,normalize_digits=False):
    """
    A basic tokenizer to tokenize text into tokens
    """    
    _DIGIT_RE = re.compile(r"\d+")  ## find digits 
    
    words = []
    tokens = list(jieba.cut(line.strip().lower()))
    if normalize_digits:
        for token in tokens:
            m = _DIGIT_RE.search(token)
            if m is None:
                words.append(token)
            else:
                words.append('_数字_')
    else:
        words = tokens 
    
    return words 

#%%
    
## same thing, for xiaohuangji data, do not need to run this, 
## already tokenized
def save_tokenized_data(train_enc,train_dec,PROCESSED_PATH):
    train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = [],[],[],[]
    save_file_path = os.path.join(PROCESSED_PATH,'processed_tokens.p')
    
    _make_dir(PROCESSED_PATH)
    
    train_enc_tokens = [_basic_tokenizer(t) for t in train_enc]
    print('Train_enc_token done.')
    
    train_dec_tokens = [_basic_tokenizer(t) for t in train_dec]
    print('Train_dec_token done.')
    
    pickle.dump((train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens),open(save_file_path,'wb'))
    
    return train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens



#%%
def main():
    data_path = "../data/universal/raw/universal.csv"
    PROCESSED_PATH = "../data/universal/processed"
    scaling_factor = 50 
    context,answers = get_lines(data_path)

    _ = save_tokenized_data(context*scaling_factor,answers*scaling_factor,PROCESSED_PATH)


if __name__ == '__main__':
  main()







