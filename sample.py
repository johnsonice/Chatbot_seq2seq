# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:22:50 2017

@author: chuang
"""
import re
import os
import gensim
from gensim.models.word2vec import Word2Vec
import time
import datetime
#%%



file_path = 'sample.txt'

with open(file_path,'r',encoding='utf-8') as f:
    file = f.readlines()

#%%

lines = [re.sub('   +','||',l.strip("\n")).split("||")[1:] for l in file]
sentences = [s.split(' ') for l in lines for s in l]
sentences = [[w for w in s if w !=''] for s in sentences]

#%%
n_dim = 300
window = 5 
downsampling = 0.001
seed = 1 
num_workers = os.cpu_count()-2    ## not sure if this is a good idea
min_count = 30 
douban_w2v = Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=n_dim,
    min_count=min_count,
    window= window,
    sample=downsampling
)
## build the vocabulary
douban_w2v.build_vocab(sentences)

#%%
## train w2v model 
corpus_count = douban_w2v.corpus_count
overall_start_time = time.time()
for i in range(200):
    start_time = time.time()
    iteration = 1000
    print('running',i+1,'-',(i+1)*iteration)
    if gensim.__version__[0] =='1':
        douban_w2v.train(sentences)
    else:
        douban_w2v.train(sentences,total_examples=corpus_count,epochs = iteration)
    
    ## save trained word2 to vect model 
    if not os.path.exists("trained"):
        os.makedirs("trained")
    file_path = os.path.join('trained','douban_'+'.w2v')
    douban_w2v.save(file_path)
    time_used = str(datetime.timedelta(seconds=time.time() - start_time))
    print('finished {} iterations, timeused {}'.format((i+1)*iteration,time_used))
