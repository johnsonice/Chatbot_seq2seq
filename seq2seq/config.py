#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:35:29 2017

@author: huang
"""
# parameters for processing the dataset
DATA_PATH = './data/xiaohuangji'
#CONVO_FILE = 'xiaohuangji50w_fenciA.conv'
#LINE_FILE = 'xiaohuangji50w_fenciA.conv'
LINE_FILE = 'xiaohuangji50w_nofenci.conv'
USER_DICT = './data_util/userdict.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = './data/xiaohuangji/processed'
CPT_PATH = 'checkpoints'
CPT_PATH_FINAL = 'checkpoints_final'
SUMMARY_PATH = 'summaries'
BUCKETS=[(10, 8), (14, 12), (19, 16), (26, 23), (43, 40),(50,50)]

#######################
## data preprocess steps
testset_size = 0.001
max_conv_length = 6


#######################
## determine structure
#######################
bidirection = False
hrnn = False
## tensorboard
tensorboard = True


#######################
## model_inputs 
#######################
epochs = 10
batch_size = 32
rnn_size = 640
attention_size = 640
# Number of Layers
num_layers = 4
if bidirection: num_layers = int(num_layers/2)
decoder_num_layers = 4 
# Embedding Size
encoding_embedding_size = 400
decoding_embedding_size = 400
keep_probability = 0.8
max_target_sentence_length= 55
beam_width = 10
source_vocab_size = 100000
target_vocab_size = 100000
# exponential learning rate decaly prams 
# Learning Rate
learning_rate = 0.001
learning_rate_decay_scheme = False
clear_step = False
start_decay_step = 100000
decay_steps = 100000
decay_factor = 0.7


###################
### display steps##
###################
display_step = 100
save_step = 10000
start_shufle = 5











