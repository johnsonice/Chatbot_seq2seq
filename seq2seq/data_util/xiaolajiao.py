# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:01:18 2017
xiaolajiao data process 
"""

import os 
import pickle 
from collections import Counter
import user_replace
import jieba
import re

#%%

# parameters for processing the dataset
DATA_PATH = '../data/xiaolajiao/raw'
USER_DICT = './userdict.txt'
PROCESSED_PATH = '../data/xiaolajiao/processed'
ENCODING = ['UTF-16','utf-8']
jieba.load_userdict(USER_DICT)

DELETE = ['\ufeff']

#%%
def replace_tokens(text,replace_dict):
    for k,v in replace_dict.items():
        pattern = re.compile("|".join(v)) 
        text = pattern.sub(k,text)
    
    pattern = re.compile("|".join(DELETE)) 
    text = re.sub(pattern,'',text)
    return text


def read_txt(file_path,encoding):
    with open(os.path.join(DATA_PATH,file_path), 'r',encoding=encoding,errors='replace') as f:
        text = f.read()
        text = replace_tokens(text,user_replace.replace_dict)
        lines = text.split('\n')
        lines = [re.split(':|：',s.strip()) for s in lines]
        lines = [s for s in lines if len(s)==2]
        combine_lines = []
        for idx,s in enumerate(lines):
            if idx == 0 :
                combine_lines.append(s)
            else:
                if s[0] == combine_lines[-1][0]:
                    combine_lines[-1][1] += ' ' + s[1] 
                else:
                    combine_lines.append(s)
        combine_lines = [s[1] for s in combine_lines]
    return combine_lines

def clear_convs(convs):
    # clear all answers with =. = 
    convs = [c for c in convs if len(c[0]) < 51 and len(c[1])< 51]

    return convs

def get_convs(lines):
    """ Get conversations from the processed line file """
    convs = []
    for idx in range(len(lines[:-1])):
        convs.append(lines[idx:idx+2])
    return convs

def context_answers(convs):
    context = [s[0] for s in convs]
    answers = [s[1] for s in convs]
    assert len(context) == len(answers)
    return context,answers

#%%
#x = read_txt('18_Füèsñ¬F«¦s+ò_yes.txt',ENCODING[0])
##%%
#convs = get_convs(x)

#%%
def main():
    convs_list = []
    data_files = os.listdir(DATA_PATH)
    for p in data_files:
        #print(p)
        try:
            combine_lines = read_txt(p,ENCODING[0])
        except:
            combine_lines = read_txt(p,ENCODING[1])
        ## eyeball mistakes, encoding issues 
        #print(combine_lines[0])
        ## convert to conversions 
        convs = get_convs(combine_lines)
        convs_list.extend(convs)
        
    convs_list_short = clear_convs(convs_list)




