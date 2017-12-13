#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:24:25 2017

@author: chengyu
"""

### chat bot inference 
import pickle
import tensorflow as tf
import data_helper as helper
import config
import os 
import numpy as np
import jieba
import re
from data_util.user_replace import custom_dict
#from nltk.tokenize import  word_tokenize 
USER_DICT = './data_util/userdict.txt'
jieba.load_userdict(USER_DICT)

#%%
dir(tf.contrib)
class chatbot(object):

    def __init__(self, config):
        vocab_path = os.path.join(config.OVERALL_PROCESSED_PATH,'vocab.p')
        self.vocab_to_int,self.int_to_vocab = helper.load_vocab(vocab_path)
        self.graph, self.sess = self.load_graph()
        self.custom_dict = custom_dict
        self.inference_logits,self.input_data,self.source_sequence_length,self.keep_prob = self.get_tensors()
        print('Chatbot model created')
        
    def load_graph(self):
        tf.reset_default_graph()
        graph = tf.Graph()
        sess = tf.Session(graph = graph)
        with graph.as_default():
            lattest_ckpt = tf.train.latest_checkpoint(config.CPT_PATH)
            if lattest_ckpt is not None:
                loader = tf.train.import_meta_graph(lattest_ckpt + '.meta')
                loader.restore(sess, lattest_ckpt)
                print("Model restored.")
            else:
                raise ValueError('no lattest ckpt found')
        
        return graph, sess
    
    def get_tensors(self):
        
        inference_logits = self.graph.get_tensor_by_name('predictions:0')
        input_data = self.graph.get_tensor_by_name('input:0')
        #target_sequence_length = graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = self.graph.get_tensor_by_name('source_sequence_length:0')
        #hrnn_sequence_length = graph.get_tensor_by_name('sequence_length:0')
        keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            
        return inference_logits,input_data,source_sequence_length,keep_prob


    def get_response(self,user_in):
        #user_in_tokens = [i for i in user_in]
        user_in_tokens = [list(jieba.cut(user_in[0]))]
        pad_encoder_input = np.array(helper.pad_answer_batch(user_in_tokens,self.vocab_to_int))
        source_lengths = [pad_encoder_input.shape[1]]*pad_encoder_input.shape[0]
        
        output = self.sess.run(
            self.inference_logits,
            {self.input_data: pad_encoder_input,
             self.source_sequence_length: source_lengths,
             self.keep_prob: 1.0})
        
        result = self.post_process(output)
        return result
    
    def replace_custom_tokens(self,res):
        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in self.custom_dict.items())
        pattern = re.compile("|".join(rep.keys()))
        res = pattern.sub(lambda m: rep[re.escape(m.group(0))], res)
        return res
    
    def post_process(self,output):
        results = list()
        
        if config.beam_width> 0 :
            output = np.squeeze(output.transpose([0,2,1]))
            end_token = self.int_to_vocab[1]
            for i in range(output.shape[0]):
                res = output[i]
                res = [self.int_to_vocab[s] for s in res if s != -1 and s !=0] ## get ride of paddings 
                if end_token in res:
                    end_idx = res.index(end_token)   ## get position of sentance end token  
                    res = res[:end_idx]
                results.append(self.replace_custom_tokens(''.join(res)))
            return(results)
        else:
            result = [self.int_to_vocab[l] for s in output for l in s if l != 0]
            if end_token in result:
                    end_idx = result.index(end_token)  ## get position of sentance end token 
                    result = result[:end_idx]
            result = self.replace_custom_tokens(''.join(result))
            return result 
    

            
#%%

## load chatbot 
chatbot = chatbot(config)

#%%
user_ins= ['我还不了解你，不知道说什么','学习我的思维？','你知道我在想什么吗？',
             '到底是怎么回事','你怎么看','其实我最大的兴趣是挣钱，还是高效率的',
             '你在做什么','听说你能陪人聊天？',
             '你能帮我卖东西么','你是做什么工作的','你是人还是机器人？']

#%%
for i in user_ins:
    user_in = [i]
    print('ask:',user_in)
    print('response:',chatbot.get_response(user_in)[0])


#%5
##%%
#user_in = ['你喜欢吃薯条吗']
#response = chatbot.get_response(user_in)
#print(response)

#%%
#
#ask = list()
#ans = list()
#for i in user_ins:
#    user_in = [i]
#    ask.append(user_in)
#    ans.append(chatbot.get_response(user_in))
#
##%%
#ask = [x[0] for x in ask]
#result = list()
#for i,e in enumerate(ans):
#    result.append((ask[i],ans[i]))
#    
##%%
#pickle.dump(result,open('results.p','wb'))
## x = pickle.load(open('results.p','rb'))
