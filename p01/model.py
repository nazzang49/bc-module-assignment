# Encoder
# Decoder
# SeqtoSeq
from utils import *
from torch import nn

import argparse
import torch
import random

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


class Seq2seq(nn.Module):
    '''
    A class of model to practice Seq2Seq with attention

    Args:
        encoder (nn.Module): encoding input data with GRU
        decoder (nn.Module): decoding encoder outputs and hidden with self attention
    '''
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
        # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L), encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)
        # (인코딩 순서) Embedding => GRU => Linear with Tanh
        encoder_outputs, hidden = self.encoder(src_batch, src_batch_lens)
        input_ids = trg_batch[:, 0]  # (B)
        batch_size = src_batch.shape[0]
        outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

        for t in range(1, trg_max_len):
            # decoder_outputs: (B, V), hidden: (1, B, d_h)
            # (디코딩 순서) Embedding => GRU => Attention => Linear with Concat
            decoder_outputs, hidden = self.decoder(input_ids, encoder_outputs, hidden)
            outputs[t] = decoder_outputs
            _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)
            
            # teacher forcing => 예측 단어 vs 정답 단어 as 디코더 다음 입력 단어
            input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

        return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='seq2seq', help='A type of model (default: skip-gram)')
    parser.add_argument('--window_size', type=int, default=2, help='A sliding window size (default: 2)')

    args = parser.parse_args()
    print(' -- Check Args -- ')
    print(args)

    try:
        if args.model_type == 'skip':
            print(' -- Make Skip-Gram Class -- ')
            train_data = get_train_data()
            w2i = make_skipgram_dataset(train_data)
            skipgram = SkipGram(vocab_size=len(w2i), dim=256) # vocab size 60 in here
            print(' [ Type of Model ] ', type(skipgram))
            print(' -- Skip-Gram Class Done -- ')
        elif args.model_type == 'seq2seq':
            print(' -- Make Seq2Seq Class -- ')
            vocab_size = 100
            src_dict, trg_dict = make_seq2seq_dataset()
            trg_max_len = trg_dict['trg_max_len']

            # TODO: call encoder and decoder
            # seq2seq = Seq2seq(encoder, decoder)
            # print(' [ Type of Model ] ', type(seq2seq))
            print(' -- Seq2Seq Class Done -- ')
        else:
            raise ValueError('[ Not Found Model ] 존재하지 않는 model_type 입니다.')
    except ValueError as e:
        print(e)