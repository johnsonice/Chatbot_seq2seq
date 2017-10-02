#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 20:28:45 2017

@author: chengyu
"""
import tensorflow as tf 
import numpy as np

## Build Seq2seq model 

## model_inputs 
# Number of Epochs
epochs = 1
# Batch Size
batch_size = 64
# RNN Size
rnn_size = 100
# Number of Layers
num_layers = 1
# Embedding Size
encoding_embedding_size = 500
decoding_embedding_size = 500
# Learning Rate
learning_rate = 0.0001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 1000
source_vocab_size = 10000
#%%
def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    # TODO: Implement Function
    
    input_data = tf.placeholder(tf.int32,[None,None],name='input')
    targets = tf.placeholder(tf.int32,[None,None],name='targets')
    lr = tf.placeholder(tf.float32,name='learning_rate')
    keep_pro = tf.placeholder(tf.float32,name='keep_prob')
    target_sequence_length = tf.placeholder(tf.int32,(None,),name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length,name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32,(None,),name='source_sequence_length')
    
    return input_data, targets, lr, keep_pro, target_sequence_length, max_target_sequence_length, source_sequence_length

input_data, targets, lr, keep_pro, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

#%%

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    # TODO: Implement Function
    # Embeding
    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs,source_vocab_size,encoding_embedding_size)
    # RNN cell 
    def make_cell(rnn_size,keep_prob):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        drop_cell = tf.contrib.rnn.DropoutWrapper(enc_cell,output_keep_prob=keep_prob)
        
        return drop_cell
    
    #enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size,keep_prob) for _ in range(num_layers)])
    enc_cell = make_cell(rnn_size,keep_prob)
    enc_output,enc_state = tf.nn.dynamic_rnn(enc_cell,enc_embed_input,sequence_length=source_sequence_length,dtype=tf.float32)
    
    return enc_output, enc_state

enc_output,enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_probability, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size)

#%%
hidden_state = enc_state[1]
#%%