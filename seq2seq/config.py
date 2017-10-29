#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:35:29 2017

@author: huang
"""
# parameters for processing the dataset
DATA_PATH = './data/xiaohuangji'
#CONVO_FILE = 'xiaohuangji50w_fenciA.conv'
LINE_FILE = 'xiaohuangji50w_fenciA.conv'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = './data/xiaohuangji/processed'
CPT_PATH = 'checkpoints'

BUCKETS=[(10, 8), (14, 12), (19, 16), (26, 23), (43, 40),(50,50)]

testset_size = 0.001
max_conv_length = 6

## model_inputs 
# Number of Epochs
epochs = 5000
# Batch Size
batch_size = 32
# RNN Size
rnn_size = 512
attention_size = 512
# Number of Layers
num_layers = 4
decoder_num_layers = 4 
# Embedding Size
encoding_embedding_size = 300
decoding_embedding_size = 300
# Learning Rate
learning_rate = 0.001


# Dropout Keep Probability
keep_probability = 0.8
display_step = 100
save_step = 1000
source_vocab_size = 100000
target_vocab_size = 100000

max_target_sentence_length= 55

beam_width = 10