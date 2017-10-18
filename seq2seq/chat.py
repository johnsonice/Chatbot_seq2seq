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
hrnn_sequence_length = graph.get_tensor_by_name('hrnn_sequence_length:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

#%%

def get_response(user_in):
    user_in_tokens = [[word_tokenize(i.lower()) for i in user_in]]
    pad_encoder_input = np.array(helper.pad_context_batch(user_in_tokens,vocab_to_int))
    source_lengths = [pad_encoder_input.shape[2]]*pad_encoder_input.shape[1]
    hrnn_lengths = [pad_encoder_input.shape[1]]
    
    output = sess.run(
        inference_logits,
        {input_data: pad_encoder_input,
         source_sequence_length: source_lengths,
         hrnn_sequence_length:hrnn_lengths,
         keep_prob: 1.0})
    
    result = [int_to_vocab[l] for s in output for l in s if l != 0]
    return result
#%%

user_in = ['You sounds very serious about this','what is going on?']
print(get_response(user_in))