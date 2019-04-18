# Generative Chatbot

## About

Uses sequence to sequence model for chatbot response generation. Includes:

- Bi-directional RNN
- Attention
- Beam Search

## Organization

- `data_util`: python scripts for data cleaning
- `scripts`: blue score and rouge score
- `chat.py`: inference (testing)
- `config.py`: all hyperparameters
- `data_helper.py`: data processing scripts related to the model
- `iter_util.py`: functions for loading data into the graph
- `seq2seq.py`: code for model architecture
- `train.py`: main script for training

## Training
Code for training the model is in `train.py`. The following inputs are needed:

- `vocab.p`: tuple of (word to freq_rank dict, freq_rank to word dict, list of words)
- `processed_tokens_clean.p`: tuple of (list of inputs, list of outputs) 

Put these in `./data/processed`. To run training:

```
python train.py
```

## Evaluation metrics

- bleu score and rouge score as implemented in Google NMT

## Additional implementations to do

- DAWnet: global, wide, and deep channels [paper and code](https://sigirdawnet.wixsite.com/dawnet)
- Topic-based evaluation: would need list of topics [paper](https://arxiv.org/pdf/1801.03622.pdf)
- Anti-language model: suppresses generic responses [code](https://github.com/Marsan-Ma-zz/tf_chatbot_seq2seq_antilm)
- Deep Transition Architecture: [paper](https://arxiv.org/pdf/1707.07631.pdf), [code](https://github.com/Avmb/deep-nmt-architectures)
- Layer normalization

## Other materials

- Chengyu's [notes](https://github.com/johnsonice/tensorflow/blob/master/notes/dataset_api.ipynb) on dataset api
- [Google NMT](https://github.com/tensorflow/nmt) 
