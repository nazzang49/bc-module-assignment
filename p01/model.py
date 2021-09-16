import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        """
        ntoken : number of tokens in dictionary
        ninp : dimension of embedding space
        nhid : dimension of hidden state
        nlayers : number of layers in RNN
        """
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers,
                              nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # dropout은 encode 후, decode 전 임의로 넣어도 무방.
        encoded = self.encoder(input)
        # encoded=self.drop(encoded)
        outs, _ = self.rnn(encoded, hidden)
        # outs=self.drop(outs)
        decoded = self.decoder(outs).view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
        query = decoder_hidden.squeeze(0)  # (B, d_h)
        # 질문? : energy 변수 설정 할 때 다시 unsqueeze를 해주는데 차라리 바로 transpose(0,1) 하면 되는 것 아닌가여?
        key = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

        energy = torch.sum(
            torch.mul(key, query.unsqueeze(1)), dim=-1)  # (B, S_L)
        """ 각 batch당 key와 query의 내적을 수행
        query : decoder_hidden 와 관련
        key, value : encoder_outputs 와 관련
        
    example:
        key   = [[[1, 2, 3, 4],
                  [2, 3, 4, 5]],
                 [[3, 4, 5, 6],
                  [4, 5, 6, 7]]] -> (B = 2, S_L = 2, d_h = 4) : 길이(토큰)이 2인 2개의 문장
        query = [[1, 0, 1, 0],
                 [0, 1, 0, 1]] -> (B = 2, d_h = 4)
        query.unsqueeze(1) = [[[1, 0, 1, 0]],
                              [[0, 1, 0, 1]]] -> (B = 2, S_L = 1, d_h = 4)
        
        mul(key, query.unsqueeze(1)) : broadcast와 Element-wise operations 로 인해 
          = [[[1, 0, 3, 0],
              [2, 0, 4, 0]],
             [[0, 4, 0, 6],
              [0, 5, 0, 7]]])

        sum(mul(key, query.unsqueeze(1)), dim = -1)
          = [[ 4,  6],
             [10, 12]]
    """
        attn_scores = F.softmax(energy, dim=-1)  # (B, S_L)
        attn_values = torch.sum(torch.mul(encoder_outputs.transpose(
            0, 1), attn_scores.unsqueeze(2)), dim=1)  # (B, d_h)
        # attn_values : weight를 준 value 결과
        return attn_values, attn_scores


class ConcatAttention(nn.Module):
    """
    [Summary]
      score를 계산하는 dot, general, concat 방법 중 concat 방법으로
      score(h_t, h_s) = v_a x tanh(W_a[h_t;h_s])
      를 구현

    Arg:
        hidden_size (int) : hidden state dimension
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.w = nn.Linear(2*hidden_size, hidden_size, bias=False)
        # W_a[h_t;h_s] 식에서 볼 수 있듯 [h_t;h_s]으로 인해 hidden_size가 2배가 된다.
        self.v = nn.Linear(hidden_size, 1, bias=False)
        # v layer를 통해 vector를 scalar로 만들어 준다

    def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
        src_max_len = encoder_outputs.shape[0]

        decoder_hidden = decoder_hidden.transpose(
            0, 1).repeat(1, src_max_len, 1)  # (B, S_L, d_h)
        # encoder_outputs의 shape과 맞춰 주기 위해 진행되는 과정

        """
        repeat : input으로 넣어준 숫자의 배수에 맞춰 크기를 키워준다.
        
        example:
            a = [[1,2,3,4,5],
                [2,3,4,5,6]]
            a.repeat(2,1) = [[1, 2, 3, 4, 5],
                            [2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5],
                            [2, 3, 4, 5, 6]])
            b = [1, 2, 3]
            b.repeat(1,2,3) = [[[1, 2, 3, 1, 2, 3, 1, 2, 3],
                                [1, 2, 3, 1, 2, 3, 1, 2, 3]]]
        """
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

        concat_hiddens = torch.cat(
            (decoder_hidden, encoder_outputs), dim=2)  # (B, S_L, 2d_h)
        # concat_hiddens = [h_t;h_s]
        energy = torch.tanh(self.w(concat_hiddens))  # (B, S_L, d_h)
        # energy = tanh(W_a[h_t;h_s])
        attn_scores = F.softmax(self.v(energy), dim=1)  # (B, S_L, 1)
        # attn_scores : v(tanh(W_a[h_t;h_s])) -> softmax로 확률로 만들어줌
        attn_values = torch.sum(
            torch.mul(encoder_outputs, attn_scores), dim=1)  # (B, d_h)
        # attn_values = attn_scores의 weight를 바탕으로 value 계산

        return attn_values, attn_scores
