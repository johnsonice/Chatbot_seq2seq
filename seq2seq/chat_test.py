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
#from nltk.tokenize import  word_tokenize 


#%%
dir(tf.contrib)
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
#hrnn_sequence_length = graph.get_tensor_by_name('sequence_length:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

#%%

def get_response(user_in):
    user_in_tokens = [i for i in user_in]
    pad_encoder_input = np.array(helper.pad_answer_batch(user_in_tokens,vocab_to_int))
    source_lengths = [pad_encoder_input.shape[1]]*pad_encoder_input.shape[0]
    results = []
    
    output = sess.run(
        inference_logits,
        {input_data: pad_encoder_input,
         source_sequence_length: source_lengths,
         keep_prob: 1.0})
    
    if config.beam_width> 0 :
        result = np.squeeze(output.transpose([0,2,1]))
        end_token = int_to_vocab[1]
        for i in range(result.shape[0]):
            res = result[i]
            res = [int_to_vocab[s] for s in res if s != -1 and s !=0] ## get ride of paddings 
            if end_token in res:
                end_idx = res.index(end_token)  ## get 
                res = res[:end_idx]
            results.append(res)
        result = ''.join(results[-1])
    else:
        result = [int_to_vocab[l] for s in output for l in s if l != 0]
        if end_token in result:
                end_idx = result.index(end_token)  ## get 
                result = result[:end_idx]
        result = ''.join(result)
    return result

#%%
user_in = ['你喜欢吃薯条']
result = get_response(user_in) 
print(result)
    
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
    
