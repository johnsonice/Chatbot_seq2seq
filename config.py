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


TESTSET_SIZE = 0.1 


## model_inputs 
# Number of Epochs
epochs = 1
# Batch Size
batch_size = 2
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
source_vocab_size = 100000
target_vocab_size = 100000

hrnn_size = 300
hrnn_num_layers = 2
hrnn_kepp_probability = 0.5


max_target_sentence_length= 50