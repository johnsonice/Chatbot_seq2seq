#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:51:20 2018

@author: chengyu
"""

### test generative evaluation model 

import chat_test
import config
import numpy as np
import os 
import csv
#%%

bot = chat_test.chatbot(config)


#%%

data_path = './data_util/xiaolajiao_answers/xiaolajiao_answers.csv'
with open(data_path, 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    ans = list(reader)
    ans = [l[0] for l in ans]
#%%

user_ins= ['你能挣钱么？','你都能做些什么','你还有些什么本事','你能干嘛','你叫什么名字','我还不了解你，不知道说什么','学习我的思维？','你知道我在想什么吗？',
             '到底是怎么回事','你怎么看','其实我最大的兴趣是挣钱，还是高效率的',
             '你在做什么','听说你能陪人聊天？',
             '你能帮我卖东西么','你是做什么工作的','你是人还是机器人？']

#%%

ans_in = ans
for ask in user_ins:
    ask_in = [ask]*len(ans_in)
    l,p = bot.evaluate(ask_in,ans_in,True)
    print("ask:",ask)
    print("answer: ", ans_in[np.argmax(p)])

#%%
for i in user_ins:
    user_in = [i]
    print('ask:',user_in)
    print('response:',bot.get_response(user_in)[0])
#index = 80
#
#ans_in = ans
#ask_in = [asks[index]]*len(ans_in)
#
#l,p = bot.evaluate(ask_in,ans_in)
#print(asks[index])
#print(np.argmax(p))
#print(ans_in[np.argmax(p)])
#print(ans_in[index])
