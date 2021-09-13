# https://colab.research.google.com/drive/1UiR_cWxO8ALkt0nBeZp9wtusWwU8NJVO
from torch.utils.data import Dataset
from utils import *
import argparse
import torch

class SkipGramDataset(Dataset):
    '''
    A dataset class for preparing skip-gram dataset to do Word2Vec

    Args:
        train_tokenized (list): 2D-Array of tokenized words from train raw sentences
        window_size (int): focus on 2-positions(default) both left and right side of current word
    '''
    def __init__(self, train_tokenized, window_size=2):
        # 초기화 시 X, Y 세팅
        self.x = []
        self.y = []

        for tokens in tqdm(train_tokenized):
            token_ids = [w2i[token] for token in tokens]
            for i, id in enumerate(token_ids):
                # token_ids 범위 내
                if i - window_size >= 0 and i + window_size < len(token_ids):
                    # target => 현위치 기준 (왼쪽 2개 단어 + 오른쪽 2개 단어)
                    self.y += (token_ids[i - window_size:i] + token_ids[i + 1:i + window_size + 1])
                    self.x += [id] * 2 * window_size

        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)

        print(' -- Original X and Y of Dataset -- ')
        print(self.x)
        print(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v_type', type=str, default='skip', help='A type of Word2Vec (default: skip-gram)')
    parser.add_argument('--window_size', type=int, default=2, help='A sliding window size (default: 2)')

    args = parser.parse_args()
    print(' -- Check Args -- ')
    print(args)

    print(' -- Prepare Skip-Gram Dataset -- ')
    train_data = get_train_data()
    train_tokenized = make_tokenized(train_data)
    word_count = get_word_count(train_tokenized)
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    w2i = get_word_to_index(word_count)

    check_train_data(train_data, train_tokenized, word_count, w2i)

    try:
        if args.w2v_type == 'skip':
            skipgram_set = SkipGramDataset(train_tokenized, args.window_size)
            print(' -- Check Skip-Gram Dataset -- ')
            print(skipgram_set[0])
        else:
            raise ValueError('[ Not Found Value ] 존재하지 않는 w2v_type 입니다.')
    except ValueError as e:
        print(e)