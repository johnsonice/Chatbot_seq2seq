#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:24:25 2017

@author: chengyu
"""

### chat bot inference 
import tensorflow as tf
import data_helper as helper
import config
import os 
import numpy as np
from nltk.tokenize import  word_tokenize 

#%%
vocab_path = os.path.join(config.PROCESSED_PATH,'vocab.p')
vocab_to_int,int_to_vocab = helper.load_vocab(vocab_path)

tf.reset_default_graph()
graph = tf.Graph()
sess = tf.Session(graph = graph)
with graph.as_default():
    lattest_ckpt = tf.train.latest_checkpoint(config.CPT_PATH)
    if lattest_ckpt is not None:
        loader = tf.train.import_meta_graph(lattest_ckpt + '.meta')
        loader.restore(sess, lattest_ckpt)
        print("Model restored.")
    else:
        raise ValueError('no lattest ckpt found')

#%%

inference_logits = graph.get_tensor_by_name('predictions:0')
input_data = graph.get_tensor_by_name('input:0')
#target_sequence_length = graph.get_tensor_by_name('target_sequence_length:0')
source_sequence_length = graph.get_tensor_by_name('source_sequence_length:0')
hrnn_sequence_length = graph.get_tensor_by_name('sequence_length:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

#%%

def get_response(user_in):
    user_in_tokens = [i for i in user_in]
    pad_encoder_input = np.array(helper.pad_answer_batch(user_in_tokens,vocab_to_int))
    source_lengths = [pad_encoder_input.shape[1]]*pad_encoder_input.shape[0]
    
    output = sess.run(
        inference_logits,
        {input_data: pad_encoder_input,
         source_sequence_length: source_lengths,
         keep_prob: 1.0})
    
    result = [int_to_vocab[l] for s in output for l in s if l != 0]
    return result
#%%

user_ins= ['你笨吗','你叫什么名字',
 '你喜欢吃炸薯条吗',
 '到底是怎么回事','你怎么看','你能做什么',
 '你在做什么','你有什么问题','你有问题吗','你打算做什么',
 '你是做什么工作的']

#%%
for i in user_ins:
    user_in = [i]
    print('ask:',user_in)
    print('response:',get_response(user_in))
    
#user_in = ['你喜欢吃薯条吗']
#print(get_response(user_in))