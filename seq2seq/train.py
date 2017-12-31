# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:29:12 2017

@author: chuang
"""
### helper.py, process data, and batch data 
import os 
import data_helper as helper
from random import shuffle
#import numpy as np
import config 
import tensorflow as tf 
#import numpy as np 
import seq2seq
#import pickle
#from tensorflow.python.layers.core import Dense
#%%
## first, load and pad data 
## load all data and vocabulary
vocab_path = os.path.join(config.OVERALL_PROCESSED_PATH,'vocab.p')
train_token_path = os.path.join(config.OVERALL_PROCESSED_PATH,'processed_tokens.p')
vocab_to_int,int_to_vocab = helper.load_vocab(vocab_path)
config.source_vocab_size = len(vocab_to_int)
config.target_vocab_size = len(vocab_to_int)
train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = helper.load_training_data(train_token_path)
#train_enc_tokens, train_dec_tokens = train_enc_tokens[:500000], train_dec_tokens[:500000]
#%%
bucket_ids = helper.bucket_training_data(train_enc_tokens,train_dec_tokens)
batches =  helper.make_batches_of_bucket_ids(bucket_ids,config.batch_size)

## get a batch of data nd pad them 
#%%
## build the network 
# create inpute place holder
input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = seq2seq.model_inputs()
# get input shape 
input_shape = tf.shape(input_data)
batch_size_t = input_shape[0]
## here source sequence length might be a problem 
with tf.variable_scope("encoder"):
    enc_output,enc_state = seq2seq.encoding_layer(tf.reverse(input_data,[-1]), config.rnn_size, config.num_layers, keep_prob, 
                       source_sequence_length, config.source_vocab_size, config.encoding_embedding_size)

#%%
## build decoder 
#max_target_sentence_length = 500 
target_vocab_to_int = vocab_to_int
## we need to process targests as well, just leave it as it is for now
dec_input = seq2seq.process_decoder_input(targets,vocab_to_int)
with tf.variable_scope("decoder"):
    training_decoder_output, inference_decoder_output = seq2seq.decoding_layer_with_attention(dec_input, enc_output,enc_state,
                                                                               source_sequence_length,target_sequence_length, config.max_target_sentence_length,
                                                                               config.rnn_size,config.decoder_num_layers, target_vocab_to_int, 
                                                                               config.target_vocab_size,batch_size_t, 
                                                                               keep_prob, config.decoding_embedding_size,config.beam_width)
    
#%%

# build cost and optimizer
training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')

if config.beam_width> 0:
    #training_logits = tf.no_op()
    inference_logits = tf.identity(inference_decoder_output.predicted_ids, name='predictions')
    scores = tf.identity(inference_decoder_output.beam_search_decoder_output.scores, name='predictions')
else:
    inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')
    
global_step = tf.Variable(0, trainable=False)
learning_rate = seq2seq._get_learning_rate_decay(global_step, config)

with tf.name_scope('optimization'):
    # Loss function    
    cost = seq2seq._compute_loss(targets,training_logits,target_sequence_length,max_target_sequence_length)

    # optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients,global_step=global_step)


## summaries for tensorboard 
tf.summary.scalar("learning_rate", learning_rate)
tf.summary.scalar("train_loss", cost)
train_summary = tf.summary.merge_all()
    
    
#%%

## training steps:
    
helper._make_dir(config.CPT_PATH)
helper._make_dir(config.CPT_PATH_FINAL)
writer = tf.summary.FileWriter(config.SUMMARY_PATH)

with tf.Session() as sess:
    saver= tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    lattest_ckpt = tf.train.latest_checkpoint(config.CPT_PATH)
    if lattest_ckpt is not None:
        saver.restore(sess, os.path.join(lattest_ckpt))
        print("Model restored.")
    else:
        print("Initiate a new model.")
    
    if config.clear_step:
        	clear_step_op = global_step.assign(0)
        	sess.run(clear_step_op)
        
    lowest_l = None
    losses = list() 
    for e in range(1,config.epochs+1):
        if e > config.start_shufle-1: shuffle(batches)  # for debuging purpose, don't randomize batches for now 
        for idx,ids in enumerate(batches,1):
            #ids = [18948, 18949, 18950, 18953, 18954, 18957, 18958, 18959]
            pad_encoder_batch,pad_decoder_batch,source_lengths,target_lengths=helper.get_batch_seq2seq(train_enc_tokens, train_dec_tokens,vocab_to_int,ids)   
            ## for debuging
            #pad_encoder_batch,pad_decoder_batch,source_lengths,target_lengths,hrnn_lengths,max_length = pickle.load(open('debug.p','rb'))
            if target_lengths[0]>config.max_target_sentence_length: continue
#            try:
            _,loss,steps,learn_r,summary = sess.run(
                    [train_op,cost,global_step,learning_rate,train_summary],
                    {input_data:pad_encoder_batch,
                     targets:pad_decoder_batch,
                     lr: config.learning_rate,      ## this part is actually not used
                     target_sequence_length:target_lengths,
                     source_sequence_length:source_lengths,
                     keep_prob:config.keep_probability}
                    )
            losses.append(loss)
            ## add statistics to summary
            writer.add_summary(summary,global_step=steps)
            
            if idx % config.display_step == 0:
                losses = losses[-100:]
                l = sum(losses)/100
                print('epoch: {}/{}, iteration: {}/{}, MA loss: {:.4f}, global_steps: {}, learning rate: {:.5f}'.format(e,config.epochs,idx,len(batches),l,steps,learn_r))

                ## check lowest loss and save when lost is the lowest 
                if lowest_l is None:
                    lowest_l = l 
                else:
                    if lowest_l > l and steps> config.start_decay_step:
                        lowest_l = l 
                        ## save the check points
                        saver.save(sess, os.path.join(config.CPT_PATH_FINAL,'hrnn_bot'),global_step =steps) 
                        print('---------------lowest loss update, check points saved -----------')
                        

            if idx % config.save_step == 0 :
                ## do not save every save step, only save lowest cost checkpoints
                
                #saver.save(sess, os.path.join(config.CPT_PATH,'hrnn_bot'),global_step =steps) 
                #print('-------------- model saved ! -------------')
                #train_ids = batches[0]
                #pad_encoder_batch,pad_decoder_batch,source_lengths,target_lengths=helper.get_batch_seq2seq(train_enc_tokens, train_dec_tokens,vocab_to_int,train_ids)
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: pad_encoder_batch,
                     source_sequence_length: source_lengths,
                     keep_prob: 1.0})
                if config.beam_width>0:
                    for i in range(config.beam_width):
                        first_res = batch_train_logits[0,:,i]
                        result = [int_to_vocab[s] for s in first_res if s != -1]
                        print(result)
                else:
                    result = [int_to_vocab[l] for s in batch_train_logits for l in s if l != 0]
                    print(result)
                    

        
