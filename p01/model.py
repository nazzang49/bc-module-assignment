# Encoder
# Decoder
# SeqtoSeq

from torch import nn
from utils import *

import argparse

class SkipGram(nn.Module):
    '''
    A class of model to practice W2V training by skip-gram method

    Args:
        vocab_size (int): A total length of vocab dict
        dim (int): A dimension after embedding (default: 256)
    '''
    def __init__(self, vocab_size, dim=256):
        super(SkipGram, self).__init__()
        # 60-dim one hot encoding => 256-dim embedding
        self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
        self.linear = nn.Linear(dim, vocab_size)

    # B: batch size, W: window size, d_w: word embedding size, V: vocab size
    def forward(self, x):
        embeddings = self.embedding(x)  # (B, d_w)
        output = self.linear(embeddings)  # (B, V)
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v_type', type=str, default='skip', help='A type of Word2Vec (default: skip-gram)')
    parser.add_argument('--window_size', type=int, default=2, help='A sliding window size (default: 2)')

    args = parser.parse_args()
    print(' -- Check Args -- ')
    print(args)

    print(' -- Make Skip-Gram Class -- ')
    train_data = get_train_data()
    train_tokenized = make_tokenized(train_data)
    word_count = get_word_count(train_tokenized)
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    w2i = get_word_to_index(word_count)

    print('[ Vocab Length ]', len(w2i)) # here 60

    try:
        if args.w2v_type == 'skip':
            skipgram = SkipGram(vocab_size=len(w2i), dim=256)
            print(' -- Check Skip-Gram Class -- ')
            print(' [ Type of Model ] ', type(skipgram))
        else:
            raise ValueError('[ Not Found Value ] 존재하지 않는 w2v_type 입니다.')
    except ValueError as e:
        print(e)