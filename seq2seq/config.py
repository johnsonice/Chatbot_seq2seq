#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:35:29 2017

@author: huang
"""
# parameters for processing the dataset
DATA_PATH = './data/cornell_movie_dialogs_corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = './data/processed'
CPT_PATH = 'checkpoints'

testset_size = 0.1
max_conv_length = 6
## model_inputs 
# Number of Epochs
epochs = 5000
# Batch Size
batch_size = 8
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 512
decoding_embedding_size = 512
# Learning Rate
learning_rate = 0.01

# Dropout Keep Probability
keep_probability = 0.8
display_step = 1000
source_vocab_size = 100000
target_vocab_size = 100000

hrnn_size = 512
hrnn_num_layers = 4
hrnn_kepp_probability = 0.8
max_target_sentence_length= 60

decoder_num_layers = 4 
