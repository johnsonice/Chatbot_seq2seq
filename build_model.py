# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:29:12 2017

@author: chuang
"""

#### Build the network 
import os 
#os.chdir('d:/usr-profiles/chuang/Desktop/Dev/Chatbot_hred')
import tensorflow as tf 
import numpy as np 
import seq2seq
from tensorflow.python.layers.core import Dense
#%%
## Build Seq2seq model 

## model_inputs 
# Number of Epochs
epochs = 1
# Batch Size
batch_size = 8
# RNN Size
rnn_size = 300
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 300
decoding_embedding_size = 300
# Learning Rate
learning_rate = 0.0001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 1000
source_vocab_size = 10000
target_vocab_size = 10000

hrnn_size = 300
hrnn_num_layers = 2
hrnn_kepp_probability = 0.5

#%%
# create inpute place holder
input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = seq2seq.model_inputs()

# get input shape 
input_shape = tf.shape(input_data)
batch_size_t = input_shape[0]

# get hidden state sequence for hrnn layer
with tf.variable_scope("encoder"):
    enc_output, enc_state,hidden_states = seq2seq.encoding_layer(input_data, rnn_size, num_layers, keep_prob, 
                       source_sequence_length, source_vocab_size, 
                       encoding_embedding_size)

# run hrnn encoding layer 
with tf.variable_scope("hrnn_encoder"):
    enc_output, enc_state = seq2seq.hierarchical_encoding_layer(hidden_states, hrnn_size, hrnn_num_layers, keep_prob, 
                       source_sequence_length)
#%%
## build decoder 

max_target_sentence_length = 500 
target_vocab_to_int = {"<GO>":0,"<EOS>":1,"something":2}
## we need to process targests as well, just leave it as it is for now
dec_input = targets
#
#with tf.variable_scope("decoder"):
training_decoder_output, inference_decoder_output = seq2seq.decoding_layer(dec_input, enc_state,
                                                                           target_sequence_length, max_target_sentence_length,
                                                                           rnn_size,num_layers, target_vocab_to_int, 
                                                                           target_vocab_size,batch_size_t, 
                                                                           keep_prob, decoding_embedding_size)



#%%
training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')

masks = tf.sequence_mask(target_sequence_length,max_target_sequence_length,dtype=tf.float32,name='masks')

with tf.name_scope('optimization'):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)
    
    # optimizer 
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


#%%
    

