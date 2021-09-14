#### Import library
from typing import List, Dict, Tuple, Sequence
from collections import Counter, defaultdict
from itertools import chain

import os
from io import open
import torch
""" 
    preprocess_tokenization import 필요
"""

tokenized = make_tokenized(data)  # 운경님 make_tokenized 함수로 tokenize 실행

#### count words
def count_word(tokenized):
    word_count = defaultdict(int)  # Key: 단어, Value: 등장 횟수

    for tokens in tokenized:  # tokens는 한 문장
        for token in tokens:  # token이 한 단위
            word_count[token] += 1  # 각 token을 기준으로 등장 횟수 count

    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)  # count한 개수를 기준으로 정렬

    return len(word_count)  # tokenized 안 word 개수 리턴



#### Dictionary class
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}  # Key: 단어, Value: idx
        self.idx2word = []  # 각 idx 위치에 해당하는 단어가 저장돼 있음

    def add_word(self, word):
        if word not in self.word2idx:  # 기존 dictionary 없는 단어라면
            self.idx2word.append(word)  # idx2word 리스트에 word append
            self.word2idx[word] = len(self.idx2word) - 1  # 추가할 때 마다 가장 마지막 idx로 idx 지정
        return self.word2idx[word]  # 단어의 idx 값 리텅

    def __len__(self):
        return len(self.idx2word)  # 사전에 저장된 단어 개수 리턴


#### Corpus class
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()  # 앞서 만든 Dictionary class 객체 생성
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # train data -> tokenize
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))  # valid data -> tokenize
        self.test = self.tokenize(os.path.join(path, 'test.txt'))  # test data -> tokenize

    def tokenize(self, path):  # word tokenization
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:  # path 파일 열어서 읽기
            for line in f:
                words = line.split() + ['<eos>']  # 한 line 안에서 split으로 음절 단위로 나누고 line 끝날 때 eos 직접 추가
                for word in words:
                    self.dictionary.add_word(word)  # dicitionary에 추가

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:  # path 파일 열어서 읽기
            idss = []
            for line in f:
                words = line.split() + ['<eos>']  # 한 line 안에서 split으로 음절 단위로 나누고 line 끝날 때 eos 직접 추가
                ids = []  # 한 line을 묶는 ids
                for word in words:
                    ids.append(self.dictionary.word2idx[word])  # dictionary에서 각 단어 idx를 찾아서 ids에 append!
                idss.append(torch.tensor(ids).type(torch.int64))  # 한 라인이 끝나면 ids를 -> int 타입으로 tensor화해서 idss에 넣기
            ids = torch.cat(idss)  # 모인 idss들을 concatenate : 1차원이 됨!(맞나?)

        return ids  # token 길이만큼의 사이즈를 가진 1차원 벡터


#### Language class
class Language(Sequence[List[str]]):
    # 직접 지정해주는 특수 token, token idx
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_IDX = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_IDX = 3
        
    def __init__(self, sentences: List[str]) -> None:
        self._sentences: List[List[str]] = [sentence.split() for sentence in sentences]  # [[sentence], ... ,[sentence]]  각 sentence 안에는 음절 기준 tokenization
        self.word2idx: Dict[str, int] = None
        self.idx2word: List[str] = None

    def build_vocab(self, min_freq: int=1) -> None:  # vocab 구성하는 함수
        SPECIAL_TOKENS: List[str] = [Language.PAD_TOKEN, Language.UNK_TOKEN, Language.SOS_TOKEN, Language.EOS_TOKEN]  # 앞에 지정했던 특수 token
        
        # 특수 token + _sentences(chain으로 각각의 sentence 안 token 쭉 꺼내옴) 단어 개수 셈 (min_freq 넘을 때만 vocab에 담아줌) -> list
        self.idx2word = SPECIAL_TOKENS + [word for word, count in Counter(chain(*self._sentences)).items() if count >= min_freq]

        # Key: 단어, Value: idx -> dict
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
    
    def set_vocab(self, word2idx: Dict[str, int], idx2word: List[str]) -> None:
        # 기존에 word2idx, idx2word 가 있을 경우 그를 vocab으로 설정해주는 함수
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def __getitem__(self, index: int) -> List[str]:
        # index를 넣으면 해당 idx의 문장을 리턴 
        return self._sentences[index]
    
    def __len__(self) -> int:
        # _sentences 안 총 문장 개수를 세어줌
        return len(self._sentences)


#### Preprocess 함수
def preprocess(
    raw_src_sentence: List[str],
    raw_tgt_sentence: List[str],
    src_word2idx: Dict[str, int],
    tgt_word2idx: Dict[str, int],
    max_len: int
) -> Tuple[List[int], List[int]]:
    """ Sentence preprocessor for neural machine translation

    Preprocess Rules:
    1. All words should be converted into their own index number by word2idx.
    2. If there is no matched word in word2idx, you should replace the word as <UNK> token.
    3. You have to use matched word2idx for each source/target language.
    4. You have to insert <SOS> as the first token of the target sentence.
    5. You have to insert <EOS> as the last token of the target sentence.
    6. The length of preprocessed sentences should not exceed max_len.
    7. If the lenght of the sentence exceed max_len, you must truncate the sentence.

    Arguments:
    raw_src_sentence -- raw source sentence without any modification
    raw_tgt_sentence -- raw target sentence without any modification 
    src_word2idx -- dictionary for source language which maps words to their unique numbers
    tgt_word2idx -- dictionary for target language which maps words to their unique numbers
    max_len -- maximum length of sentences

    Return:
    src_sentence -- preprocessed source sentence
    tgt_sentence -- preprocessed target sentence

    """
    # Special tokens, use these notations if you want
    UNK = Language.UNK_TOKEN_IDX
    SOS = Language.SOS_TOKEN_IDX
    EOS = Language.EOS_TOKEN_IDX

    ### 아래에 코드 빈칸(None)을 완성해주세요
    src_sentence = []
    tgt_sentence = []
    for word in raw_src_sentence:
        if word in src_word2idx: # src dictionary에 현재의 word가 있는 경우
            src_sentence.append(src_word2idx[word])
        else:
            src_sentence.append(UNK) # src dictionary에 현재의 word가 없는 경우
    
    for word in raw_tgt_sentence:
        if word in tgt_word2idx: # tgt dictionary에 현재의 word가 있는 경우
            tgt_sentence.append(tgt_word2idx[word])
        else:
            tgt_sentence.append(UNK) # tgt dictionary에 현재의 word가 없는 경우

    src_sentence = src_sentence[:max_len] # max_len까지의 sequence만
    tgt_sentence = [SOS] + tgt_sentence[:max_len-2] + [EOS] # SOS, EOS token을 추가하고 max_len까지의 sequence만

    return src_sentence, tgt_sentence