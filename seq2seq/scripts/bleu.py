#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:47:25 2018

@author: chengyu
"""
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import os, sys
from data_util import user_replace
import jieba
USER_DICT = './data_util/userdict.txt'
jieba.load_userdict(USER_DICT)
import re

#%%

def replace_tokens(text,replace_dict):
    for k,v in replace_dict.items():
        pattern = re.compile("|".join(v)) 
        text = pattern.sub(k,text)
        
    return text

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

def calculate_bleu(reference,candidate):
    score_list = []
    assert len(reference) == len(candidate)
    
    for i in range(len(reference)):
        s = reference[i]
        c = candidate[i]
        sc = sentence_bleu(s,c)
        score_list.append(sc)
        
    return score_list,sum(score_list)/len(score_list)

def calculate_bleu_from_lists(reference,candidate):
    r_list = []
    c_list = []
    
    for r in reference:
        r = replace_tokens(r,user_replace.replace_dict)
        r = _basic_tokenizer(r)
        r_list.append([r])
        
    for c in candidate:
        c = replace_tokens(c,user_replace.replace_dict)
        c = _basic_tokenizer(c)
        c_list.append(c)
    
    score_list,avg_score = calculate_bleu(r_list,c_list)
    
    return score_list,avg_score
#%%
reference = ['我是一只猪','我是一只猪']
candidate = ['我是一只猪','我是']

score_list,avg_score = calculate_bleu_from_lists(reference,candidate)

#def tokenize_reference_sentence(sentence):
    
