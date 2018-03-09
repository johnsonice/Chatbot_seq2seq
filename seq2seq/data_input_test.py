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
#from random import shuffle
import numpy as np
import config 
import tensorflow as tf 
#import numpy as np 
#import seq2seq
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
def get_iterator(enc_dataset,dec_dataset,src_vocab_table,tgt_vocab_table,
                 config,output_buffer_size=None):
    
    ## define output buffer_size
    if not output_buffer_size:
        output_buffer_size = config.batch_size*1000
    
    ## generate source data generator
    def src_generator():
        seq = np.array(enc_dataset)
        for el in seq:
            yield el
    ## generate target data generator
    def tgt_generator():
        seq = np.array(dec_dataset)
        for el in seq:
            yield el
    ## 
    src_dataset = tf.data.Dataset().from_generator(src_generator,
                                           output_types=tf.int32, 
                                           output_shapes=tf.TensorShape([None]))
    #src_dataset = src_dataset.map(lambda words: (words,tf.size(words)))
    
    tgt_dataset = tf.data.Dataset().from_generator(tgt_generator,
                                               output_types=tf.int32, 
                                               output_shapes=tf.TensorShape([None]))
    #tgt_dataset = tgt_dataset.map(lambda words: (words,tf.size(words)))
    
    ## combine src and tgt 
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    
    ## shuffle dataset
    src_tgt_dataset = src_tgt_dataset.shuffle(buffer_size=output_buffer_size,reshuffle_each_iteration=True)
    
    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    
    if config.src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src, tgt[:config.tgt_max_len]))
    
    if config.tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt: (src, tgt[:config.tgt_max_len]))
                
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src,tf.size(src),tgt,tf.size(tgt)))
    
    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            config.batch_size,                                      # batch size
            padded_shapes=(tf.TensorShape([None]),  # source vectors of unknown size
                            tf.TensorShape([]),     # size(source)
                           tf.TensorShape([None]),  # target vectors of unknown size
                            tf.TensorShape([])),    # size(target)
            padding_values=(vocab_to_int['<EOS>'],  # source vectors padded on the right with src_eos_id
                             vocab_to_int['<PAD>'],          # size(source) -- unused
                            vocab_to_int['<EOS>'],  # target vectors padded on the right with tgt_eos_id
                             vocab_to_int['<PAD>']))         # size(target) -- unused

    if config.num_buckets > 1:
        
        ## define a func to identify bucket id
        def key_func(unused_1,src_len,unused_2,tgt_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if config.src_max_len:
                bucket_width = (config.src_max_len + config.num_buckets - 1) // config.num_buckets
            else:
                bucket_width = 10
            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            
            return tf.to_int64(tf.minimum(config.num_buckets, bucket_id))
        
        ## define a wraper function
        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)
        
        ## now we can starting bucketing and butching together
        batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, 
                                                window_size=config.batch_size)
                )
    else:
        ## if num_buckets not bigger than 1 
        batched_dataset = batching_func(src_tgt_dataset)
    
    #########################################
    ## now construct returned iter object ###
    #########################################
    
    return batched_dataset

#%%
batched_dataset = get_iterator(train_enc_num, train_dec_num,vocab_to_int,vocab_to_int,config)
batched_iterator = batched_dataset.make_one_shot_iterator()
#%%
source, source_lengths, target, target_lengths = batched_iterator.get_next()
#%%

with tf.Session() as sess:
    for x in range(1):
        _ = sess.run([source,source_lengths])
        print(_) # output: [ 0.42116176  0.40666069]
        
        
    
##%%
#def enc_generator():
#    seq = np.array(train_enc_num)
#    for el in seq:
#        yield el
#
#def dec_generator():
#    seq = np.array(train_dec_num)
#    for el in seq:
#        yield el
#
#enc_dataset = tf.data.Dataset().from_generator(enc_generator,
#                                           output_types=tf.int32, 
#                                           output_shapes=tf.TensorShape([None]))
#
#enc_dataset = enc_dataset.map(lambda words: (words,tf.size(words)))
#
#dec_dataset = tf.data.Dataset().from_generator(dec_generator,
#                                           output_types=tf.int32, 
#                                           output_shapes=tf.TensorShape([None]))
#
#dec_dataset = dec_dataset.map(lambda words: (words,tf.size(words)))
#
#source_target_dataset = tf.data.Dataset.zip((enc_dataset, dec_dataset))
#
#batched_dataset = source_target_dataset.padded_batch(
#        100000,                                      # batch size
#        padded_shapes=((tf.TensorShape([None]),  # source vectors of unknown size
#                        tf.TensorShape([])),     # size(source)
#                       (tf.TensorShape([None]),  # target vectors of unknown size
#                        tf.TensorShape([]))),    # size(target)
#        padding_values=((vocab_to_int['<EOS>'],  # source vectors padded on the right with src_eos_id
#                         vocab_to_int['<PAD>']),          # size(source) -- unused
#                        (vocab_to_int['<EOS>'],  # target vectors padded on the right with tgt_eos_id
#                         vocab_to_int['<PAD>'])))         # size(target) -- unused
#
#
##%%
#batched_iterator = batched_dataset.make_one_shot_iterator()
##%%
#((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()
##%%
#
#with tf.Session() as sess:
#    for x in range(100):
#        x = sess.run([source,source_lengths])
#        print() # output: [ 0.42116176  0.40666069]
    
    
    
    
    
    
    
    
    
    
    
    