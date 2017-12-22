# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:54:08 2017

@author: chuang
"""

import os 
import pickle 
#from collections import Counter
#import user_replace
import jieba
import re
from multiprocessing import Pool 

#%%

# parameters for processing the dataset
DATA_PATH = '../data/weibo_single/raw'
USER_DICT = './userdict.txt'
PROCESSED_PATH = '../data/weibo_single/processed'
ENCODING = 'utf-8'
jieba.load_userdict(USER_DICT)

DELETE = ['\[.*?\]','\u200b']
MULTI = True

#%%
def replace_tokens(text,replace_dict=None):
#    for k,v in replace_dict.items():
#        pattern = re.compile("|".join(v)) 
#        text = pattern.sub(k,text)
    
    pattern = re.compile("|".join(DELETE)) 
    text = re.sub(pattern,'',text)
    return text

def read_txt(file_path,encoding):
    with open(os.path.join(DATA_PATH,file_path), 'r',encoding=encoding,errors='replace') as f:
        text = f.read()
        
        text = replace_tokens(text) #,user_replace.replace_dict
        convs = text.split('\n\n')
        lines = [c.split('\n') for c in convs]
        lines = [[i.strip() for i in c if i != ''] for c in lines] ## get ride of empties sentences
        lines = [c for c in lines if len(c)>1]
    return lines

def context_answers(convos):
    context,answers = [],[]
    for convo in convos:
        for index,line in enumerate(convo[:-1]):
            context.append(line)
            answers.append(convo[index+1])
        
    assert len(context) == len(answers)
    return context,answers

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

def _tokenized_data(context,answers):
    
    train_enc_tokens = [_basic_tokenizer(t) for t in context]
    print('Train_enc_token done.')
    
    train_dec_tokens = [_basic_tokenizer(t) for t in answers]
    print('Train_dec_token done.')
    
    return train_enc_tokens, train_dec_tokens

def _filter(ask_sent,answer_sent):
    
    if len(ask_sent)<3 or len(answer_sent)<2:
        return False 
    
    if "@" in ask_sent or "@" in answer_sent:
        return False
    
    return True

def filter_data(context,answers):
    '''
        filter some answer that is too short or has @ in it 
    '''
    context_return, answers_return = [],[]
    for i in range(len(context)):
        c = context[i]
        a = answers[i]
        c_sent = " ".join(c)
        a_sent = " ".join(a)
        
        if _filter(c_sent,a_sent):
            context_return.append(c)
            answers_return.append(a)
    
    return context_return,answers_return

def save_tokenized_data(train_enc_tokens,train_dec_tokens,save_file_name):
    save_file_path = os.path.join(PROCESSED_PATH,save_file_name)
    pickle.dump((train_enc_tokens, train_dec_tokens,[],[]),open(save_file_path,'wb'))
    print('Data saved')
    
    
    
    
#%%
if __name__ == "__main__":
    
    data_files = os.listdir(DATA_PATH)  ## just do two files for now, too many data 
    #%%
    asks,ans = [],[]
    for idx,file_path in enumerate(data_files):
        #file_path = 'multi_1_4.data'
        convos = read_txt(file_path,ENCODING)
        context,answers = context_answers(convos)
        
        asks.extend(context)
        ans.extend(answers)
        print('finish {}'.format(file_path))
        print('Total length {}'.format(len(asks)))
    #%%
    if MULTI:
        print('tokanizing, multi process')
        cores = os.cpu_count()-2 
        p = Pool(cores)
        context = p.map(_basic_tokenizer,asks)
        print('Finish tokenizing ask sentences')
        answers = p.map(_basic_tokenizer,ans)
        print('Finish tokenizing answer sentences')
        p.close()
        p.join()
    else:
        context,answers = _tokenized_data(asks,ans)     
    
    print("Total lentgh after tokenization: {}".format(len(context)))
    #%%
    context,answers = filter_data(context,answers)
    print("Total lentgh after filtering: {}".format(len(context)))
    #%%
    ## save into pickles
    save_tokenized_data(context,answers,'processed_tokens.p')
        
    #%%
    #print(context[:50])
    #print(answers[:50])



