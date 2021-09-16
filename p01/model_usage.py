import data
from text.batchify import padding_data

import torch
from torch import nn

'''
RNN 모델 사용법
'''

vocab_size = 100   # 사전에 정의된 단어 갯수
embedding_size = 256  # vocab_size를 임베딩한 후 차원

# 배치화하기
data = data.dataforRNN
batch, batch_lens = padding_data(data) # (B, L): (문장, padding된 문장 길이), (B) : 원래 문장의 길이
print('배치 크기')
print(batch.shape, batch_lens.shape)

# 배치 임베딩하기
embedding = nn.Embedding(vocab_size, embedding_size)
batch_emb = embedding(batch)  # (B, L, d_w)
print('임베딩 후 배치 크기')
print(batch_emb.shape)

# rnn 모델
hidden_size = 512  # RNN의 hidden size
num_layers = 1  # 쌓을 RNN layer의 개수
num_dirs = 1  # 1: 단방향 RNN, 2: 양방향 RNN

# 선언
rnn = nn.RNN(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True if num_dirs > 1 else False
)

# 모델 내의 hidden states 벡터 초기값 설정
h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)


# hidden_states: 각 time step에 해당하는 hidden state들의 묶음.
# h_n: 모든 sequence를 거치고 나온 마지막 hidden state.
hidden_states, h_n = rnn(batch_emb.transpose(0, 1), h_0)


# d_h: hidden size, num_layers: layer 개수, num_dirs: 방향의 개수
print('rnn 출력 후 결과')
print(hidden_states.shape)  # (L, B, d_h)
print(h_n.shape)  # (num_layers*num_dirs, B, d_h) = (1, B, d_h)

