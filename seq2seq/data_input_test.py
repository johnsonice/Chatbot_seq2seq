# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 08:33:05 2018

@author: chuang
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:29:12 2017

@author: chuang
"""
### helper.py, process data, and batch data 
import os 
import data_helper as helper
from random import shuffle
import numpy as np
import config 
import tensorflow as tf 
#import numpy as np 
import seq2seq
#import pickle
#from tensorflow.python.layers.core import Dense

test_size = 1000

#%%
## first, load and pad data 
## load all data and vocabulary
vocab_path = os.path.join(config.OVERALL_PROCESSED_PATH,'vocab.p')
train_token_path = os.path.join(config.OVERALL_PROCESSED_PATH,'processed_tokens_clean.p')
vocab_to_int,int_to_vocab = helper.load_vocab(vocab_path)
config.source_vocab_size = len(vocab_to_int)
config.target_vocab_size = len(vocab_to_int)
train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = helper.load_training_data(train_token_path)
if config.training_size is None:
    pass
else:
    train_enc_tokens, train_dec_tokens = train_enc_tokens[config.start_point:config.start_point+config.training_size], train_dec_tokens[config.start_point:config.start_point+config.training_size]

#%%
## convert text to token ids  

def convert_to_num(tokens_list,vocab_to_int):
    res = [helper.sentence2id(t,vocab_to_int) for t in tokens_list]
    assert len(res) == len(tokens_list)
    return res

train_enc_num,train_dec_num = convert_to_num(train_enc_tokens,vocab_to_int),convert_to_num(train_dec_tokens,vocab_to_int)


#%%

def enc_generator():
    seq = np.array(train_enc_num)
    for el in seq:
        yield el

def dec_generator():
    seq = np.array(train_dec_num)
    for el in seq:
        yield el

enc_dataset = tf.data.Dataset().from_generator(enc_generator,
                                           output_types=tf.int32, 
                                           output_shapes=tf.TensorShape([None]))

enc_dataset = enc_dataset.map(lambda words: (words,tf.size(words)))

dec_dataset = tf.data.Dataset().from_generator(dec_generator,
                                           output_types=tf.int32, 
                                           output_shapes=tf.TensorShape([None]))

dec_dataset = dec_dataset.map(lambda words: (words,tf.size(words)))

source_target_dataset = tf.data.Dataset.zip((enc_dataset, dec_dataset))

batched_dataset = source_target_dataset.padded_batch(
        2,                                      # batch size
        padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
                        tf.TensorShape([])),     # size(source)
                       (tf.TensorShape([None]),  # target vectors of unknown size
                        tf.TensorShape([]))),    # size(target)
        padding_values=((vocab_to_int['<EOS>'],  # source vectors padded on the right with src_eos_id
                         vocab_to_int['<PAD>']),          # size(source) -- unused
                        (vocab_to_int['<EOS>'],  # target vectors padded on the right with tgt_eos_id
                         vocab_to_int['<PAD>'])))         # size(target) -- unused


#%%
batched_iterator = batched_dataset.make_one_shot_iterator()
#%%
((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()
#%%

with tf.Session() as sess:
    for x in range(5):
        print(sess.run([source,source_lengths])) # output: [ 0.42116176  0.40666069]
