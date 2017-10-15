#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:35:15 2017

@author: huang
"""

### helper.py, process data, and batch data 
import os 
import random 
import re 
import numpy as np
import config 
import pickle 
from collections import Counter
from nltk.tokenize import  word_tokenize #,sent_tokenize


#%%
#########################################
## process cornell movie - dialogs data #
#########################################

## get all sentences with sentence id 
def get_lines():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    with open(file_path, 'r',encoding='utf-8',errors='replace') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line

#id2line = get_lines()

#%%

## get all conversion with sentence id in a list
def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos

convos = get_convos()
#%%
def context_answers(convos,id2line):
    context,answers = [],[]
    for convo in convos:
        for index,line in enumerate(convo[:-1]):
            context.append([id2line[line] for line in convo[:index+1]])
            answers.append(id2line[convo[index+1]])
        
    assert len(context) == len(answers)
    return context,answers

#context,answers =  context_answers(convos,id2line)

#%%
def _make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
            

def train_test_split(context,answers):
    """
    devide dateset into training and test sets 
    """
    print("Saving to txt pickle")
    # create directory to hold processed data
    _make_dir(config.PROCESSED_PATH)
    
    # random convos to create test set 
    total_numbers = len(context)
    test_size = int(total_numbers * config.testset_size)
    test_ids = random.sample([i for i in range(total_numbers)],test_size)
    
    train_enc, train_dec, test_enc,test_dec = [],[],[],[]
    for i in range(total_numbers):
        if i in test_ids:
            test_enc.append(context[i])
            test_dec.append(answers[i])
        else:
            train_enc.append(context[i])
            train_dec.append(answers[i])
        
        if i % 10000 == 0 : print('Finishing: ',i)
    
    save_file_path = os.path.join(config.PROCESSED_PATH,'processed_text.p')
    pickle.dump((train_enc, train_dec, test_enc,test_dec),open(save_file_path,'wb'))
    
    return train_enc, train_dec, test_enc,test_dec
    
#_ = train_test_split(context,answers)
#%%
def _basic_tokenizer(line,normalize_digits=True):
    """
    A basic tokenizer to tokenize text into tokens
    """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    
    _DIGIT_RE = re.compile(r"\d+")  ## find digits 
    
    words = []
    tokens = word_tokenize(line.strip().lower())
    if normalize_digits:
        for token in tokens:
            m = _DIGIT_RE.search(token)
            if m is None:
                words.append(token)
            else:
                words.append('#')
    else:
        words = tokens 
    
    return words 

#%%
def save_tokenized_data(text_pickle_path):
    train_enc, train_dec, test_enc,test_dec = pickle.load(open(text_pickle_path,'rb'))
    train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = [],[],[],[]
    save_file_path = os.path.join(config.PROCESSED_PATH,'processed_tokens.p')
    
    for t in train_enc:
        enc_convo = [_basic_tokenizer(i) for i in t]
        train_enc_tokens.append(enc_convo)
    print('Train_enc_token done.')
    
    for t in train_dec:
        enc_convo = _basic_tokenizer(t)
        train_dec_tokens.append(enc_convo)
    print('Train_dec_token done.')
    
    for t in test_enc:
        enc_convo = [_basic_tokenizer(i) for i in t]
        test_enc_tokens.append(enc_convo)
    print('Test_enc_token done.')
    
    for t in test_dec:
        enc_convo = _basic_tokenizer(t)
        test_dec_tokens.append(enc_convo)
    print('Test_dec_token done.')
    
    pickle.dump((train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens),open(save_file_path,'wb'))
    
    return train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens

### load processed_text and save processed_tokenize
#_ = save_tokenized_data(os.path.join(config.PROCESSED_PATH,'processed_text.p'))
    
#%%
########################
## Now build vocabulary 
########################
CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

## a recursive function to flatten nested lists 
def _flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in _flatten(i):
                yield j
        else:
            yield i

## so idealy, we want to drop those words that does not happend very often 
def build_vocab(pickle_file_path,CODES):
    tokens = pickle.load(open(pickle_file_path,'rb'))
    all_words = []
    for t in tokens:
        all_words.extend(list(_flatten(t)))
    
    counts = Counter(all_words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, len(CODES))}  # enumerate start from len(CODES)
    vocab_to_int = dict(vocab_to_int,**CODES)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
    
    save_file_path = os.path.join(config.PROCESSED_PATH,'vocab.p')
    pickle.dump((vocab_to_int,int_to_vocab),open(save_file_path,'wb'))
    
    return vocab_to_int,int_to_vocab

#vocab_to_int,int_to_vocab = build_vocab(os.path.join(config.PROCESSED_PATH,'processed_tokens.p'),CODES)

#%%
    
def load_vocab(vocab_path):
    vocab_to_int,int_to_vocab = pickle.load(open(vocab_path,'rb'))
    return vocab_to_int,int_to_vocab

def load_training_data(train_token_path):
    train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens= pickle.load(open(train_token_path,'rb'))
    return train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens

#vocab_path = os.path.join(config.PROCESSED_PATH,'vocab.p')
#train_token_path = os.path.join(config.PROCESSED_PATH,'processed_tokens.p')
#vocab_to_int,int_to_vocab = load_vocab(vocab_path)
#train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = load_training_data(train_token_path)

#%%
def sentence2id(line,vocab_to_int):
    return [vocab_to_int.get(token,vocab_to_int['<UNK>']) for token in line] + [vocab_to_int['<EOS>']]


#test = ['i','am','a','research','assistant']
#ids = sentence2id(test,vocab_to_int)

#%%
def pad_context_batch(batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(s) for conv in batch for s in conv])
    pad_batch = []
    for conv in batch:
        pad_sentence = [sentence2id(sentence,vocab_to_int) + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in conv]
        pad_batch.append(pad_sentence)
        
    return pad_batch

def pad_answer_batch(batch, vocab_to_int):
    max_sentence = max([len(s) for s in batch ])
    pad_batch = [sentence2id(sentence,vocab_to_int) + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in batch]

    return pad_batch

def get_batch(train_enc_tokens, train_dec_tokens,vocab_to_int,ids):
    encoder_input = [train_enc_tokens[i] for i in ids]
    pad_encoder_input = np.array(pad_context_batch(encoder_input,vocab_to_int))
    pad_encoder_shape = pad_encoder_input.shape
    decoder_input = [train_dec_tokens[i] for i in ids]
    pad_decoder_input = np.array(pad_answer_batch(decoder_input,vocab_to_int))
    pad_decoder_shape = pad_decoder_input.shape

    source_sequence_length = [pad_encoder_shape[2]]*(pad_encoder_shape[0]*pad_encoder_shape[1])
    hrnn_sequence_length = [pad_encoder_shape[1]]*pad_encoder_shape[0]
    target_sequence_length = [pad_decoder_shape[1]]*pad_decoder_shape[0]
    
    return pad_encoder_input, pad_decoder_input, source_sequence_length,target_sequence_length,hrnn_sequence_length
    
#encoder_input,decoder_input =  get_batch(train_enc_tokens, train_dec_tokens,batch_size = 1,context_length=2)
#pad_encoder_batch = pad_context_batch(encoder_input,vocab_to_int)
#pad_decoder_batch = pad_answer_batch(decoder_input,vocab_to_int)

#%%
### put training data into buckets 
def bucket_training_data(train_enc_tokens,MAX_CONV_LENGTH):
    buckets = range(1,MAX_CONV_LENGTH+1) 
    bucket_ids = {'bucket_'+str(i):list() for i in buckets}
    for i in range(len(train_enc_tokens)):
        conv_length = len(train_enc_tokens[i])
        if conv_length < MAX_CONV_LENGTH:
            bucket_id= 'bucket_' +  str(conv_length)
            bucket_ids[bucket_id].append(i)

    return bucket_ids   

def make_batches_of_bucket_ids(bucket_ids,batch_size):
    id_batches = list() 
    for i,v in bucket_ids.items():
        for batch_i in range(len(v)//batch_size):
            start_i = batch_i * batch_size
            ## slice the right amount for the batch 
            id_batches.append(v[start_i:start_i+batch_size])
    
    return id_batches
        
        
        
#%%

def main():
    ## process data and build vocabulary
    id2line = get_lines()
    context,answers =  context_answers(convos,id2line)
    _ = train_test_split(context,answers)                                            ## generage train and test data sets, save them in pickle file prcessed_text.p
    _ = save_tokenized_data(os.path.join(config.PROCESSED_PATH,'processed_text.p'))  ## generate tokenized datasets 
    _ = build_vocab(os.path.join(config.PROCESSED_PATH,'processed_tokens.p'),CODES)  ## take processed tokens, to generate dictionary


if __name__ == '__main__':
  main()
  
  
### load all data and vocabulary
#vocab_path = os.path.join(config.PROCESSED_PATH,'vocab.p')
#train_token_path = os.path.join(config.PROCESSED_PATH,'processed_tokens.p')
#vocab_to_int,int_to_vocab = load_vocab(vocab_path)
#train_enc_tokens, train_dec_tokens, test_enc_tokens,test_dec_tokens = load_training_data(train_token_path)
#
### get a batch of data nd pad them 
#encoder_input,decoder_input =  get_batch(train_enc_tokens, train_dec_tokens,batch_size = 1,context_length=2)
#pad_encoder_batch = pad_context_batch(encoder_input,vocab_to_int)
#pad_decoder_batch = pad_answer_batch(decoder_input,vocab_to_int)