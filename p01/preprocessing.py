#### Import library
from typing import List, Dict, Tuple, Sequence
from collections import Counter, defaultdict
from itertools import chain
from konlpy import tag
from tqdm import tqdm
from transformers import BertTokenizer
import argparse
import spacy  # spacy version == 3.1.2, wrote in requirements.txt
import os
from io import open
import torch

class EnglishPreprocess:
    """
    Preprocess English Texts using spacy
    It tokenize text based mostly on space ' ' and punctuation marks
    's is treated as an individual token
    Then, it'll delete stopwords and punctuation marks then return
    lemmatized data

    usage
    ep = EnglishPreprocess(text)
    preprocessed_text = ep.get_preprocessed_data()

    if you haven't installed en_core_web_sm
    run this command on Terminal

    $ python -m spacy download en_core_web_sm
    """

    def __init__(self, text):
        self.nlp = spacy.load('en_core_web_sm')
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.text = self.nlp(text)

    def is_token_allowed(self, token):
        """
        check if the token is punctuation mark or stopword
        return True only if the token is not punctuation mark and stopword
        """
        if token.is_stop or token.is_punct:
            return False
        return True

    def preprocess_token(self, token):
        """
        return lowercased lemmatized token
        """
        return token.lemma_.strip().lower()

    def get_preprocessed_data(self):
        filtered_tokens = [self.preprocess_token(
            token) for token in self.text if self.is_token_allowed(token)]
        return filtered_tokens


def characterwise_tokenizer(sentence):
    syllables = []
    for word in sentence:
        for syllable in word:
            syllables.append(syllable)
        return syllables


# Tokenizer dictionary
tokenizer = {
    'okt': tag.Okt().morphs,  # 한국어 토크나이저
    'mecab': tag.Mecab().morphs,  # 한국어 토크나이저
    'character': characterwise_tokenizer,  # 글자별 토크나이저
    'spacy': spacy.load('en_core_web_sm'),
    'bert': BertTokenizer.from_pretrained('bert-base-cased').tokenize,
}

# 테스트용 데이터
data = {'ko': [
    "정말 맛있습니다. 추천합니다.",
    "기대했던 것보단 별로였네요.",
    "다 좋은데 가격이 너무 비싸서 다시 가고 싶다는 생각이 안 드네요.",
    "완전 최고입니다! 재방문 의사 있습니다.",
    "음식도 서비스도 다 만족스러웠습니다.",
    "위생 상태가 좀 별로였습니다. 좀 더 개선되기를 바랍니다.",
    "맛도 좋았고 직원분들 서비스도 너무 친절했습니다.",
    "기념일에 방문했는데 음식도 분위기도 서비스도 다 좋았습니다.",
    "전반적으로 음식이 너무 짰습니다. 저는 별로였네요.",
    "위생에 조금 더 신경 썼으면 좋겠습니다. 조금 불쾌했습니다."
],
    'en': [
    'This assignment is about Natural Language Processing.',
],
    'mixed': [
    'S를 element가 하나인 집합은 닫힌 집합인 topological space X의 부분 집합이라고 하자.'
]
}


def make_tokenized(args):
    tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

    for sentence in tqdm(data[args.data]):
        tokens = tokenizer[args.tokenizer](sentence)
        tokenized.append(tokens)

    return tokenized


def count_word(tokenized: List[str]) -> int:
    """
    Count unique words in tokenized data

    Args:
        tokenized (list): "tokenized" data produced after make_tokenized(raw_data)
    """
    word_count = defaultdict(int)  # Key: word, Value: count

    for tokens in tokenized:
        for token in tokens:
            word_count[token] += 1

    return len(word_count)
 

class Dictionary(object):
    '''
    A Dictionary class needed to be preceded to make Corpus Class.
    You can generate Dictionary based on the data set.
    This dictionary is comprised of unique token mapping to indicies.   
    '''
    def __init__(self):
        '''
        Build word2idx (dict) and idx2word (list)
        
        word2idx (dict): {word (str) : idx (int)}
        idx2word (list): [word (str)]
        '''
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str):
        '''
        Add new word to the dictionary
        '''
        if word not in self.word2idx:  # 기존 dictionary 없는 단어라면
            self.idx2word.append(word)  # idx2word 리스트에 word append
            self.word2idx[word] = len(self.idx2word) - 1  # 추가할 때 마다 가장 마지막 idx로 idx 지정
        return self.word2idx[word]  # 단어의 idx 값 리턴

    def __len__(self):
        '''
        Count unique words in dictionary
        '''
        return len(self.idx2word)  # 사전에 저장된 단어 개수 리턴


#### Corpus class
class Corpus(object):
    '''
    Corpus class for constructing Language model

    Class including the following 
        - Load data
        - Build dictionary
        - Tokenize data based on the dictionary
        - convert tokenized output to id by tensorfying it

    '''
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        '''
        Tokenizes a text file
        '''
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:  # path 파일 열어서 읽기
            for line in f:
                words = line.split() + ['<eos>']  # 한 line 안에서 split으로 어절 단위로 나누고 line 끝날 때 eos 직접 추가  # 이삭님 tokenization 함수 추가
                for word in words:
                    self.dictionary.add_word(word)  # dicitionary에 추가

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:  # path 파일 열어서 읽기
            idss = []
            for line in f:
                words = line.split() + ['<eos>']  # 한 line 안에서 split으로 어절 단위로 나누고 line 끝날 때 eos 직접 추가
                ids = []  # 한 line을 묶는 ids
                for word in words:
                    ids.append(self.dictionary.word2idx[word])  # dictionary에서 각 단어 idx를 찾아서 ids에 append!
                idss.append(torch.tensor(ids).type(torch.int64))  # 한 라인이 끝나면 ids를 -> int 타입으로 tensor화해서 idss에 넣기
            ids = torch.cat(idss)  # 모인 idss들을 concatenate : 1차원이 됨!

        return ids  # token 길이만큼의 사이즈를 가진 1차원 벡터 리턴


class Language(Sequence[List[str]]):
    '''
    Language class for constructing Language model
    This class is needed when the unif of interset is a sentence, not a word, 
    like nueral machine translation task.

    Class including the following 
        - Map special tokens to indicies
        - Build vocab(dictionary) or Set vocab(dictionary)
        - Preprocess sentences
    '''

    # Special tokens
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_IDX = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_IDX = 3
        
    def __init__(self, sentences: List[str]) -> None:
        self._sentences: List[List[str]] = [sentence.split() for sentence in sentences]  # [[sentence], ... ,[sentence]]  각 sentence 안에는 어절 기준 tokenization
        self.word2idx: Dict[str, int] = None
        self.idx2word: List[str] = None

    def build_vocab(self, min_freq: int=1) -> None:
        '''
        Build vocabulary
        '''
        SPECIAL_TOKENS: List[str] = [Language.PAD_TOKEN, Language.UNK_TOKEN, Language.SOS_TOKEN, Language.EOS_TOKEN]
        
        # 특수 token + _sentences(chain으로 각각의 sentence 안 token 쭉 꺼내옴) 단어 개수 셈 (min_freq 넘을 때만 vocab에 담아줌) -> list
        self.idx2word = SPECIAL_TOKENS + [word for word, count in Counter(chain(*self._sentences)).items() if count >= min_freq]

        # Key: 단어, Value: idx -> dict
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
    
    def set_vocab(self, word2idx: Dict[str, int], idx2word: List[str]) -> None:
        '''
        Set other vocabulary as vocab of this class
        '''
        self.word2idx = word2idx
        self.idx2word = idx2word

    def preprocess(
        self,
        raw_src_sentence: List[str],
        raw_tgt_sentence: List[str],
        src_word2idx: Dict[str, int],
        tgt_word2idx: Dict[str, int],
        max_len: int
    ) -> Tuple[List[int], List[int]]:
        '''Sentence preprocessor for neural machine translation

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

        '''
        # Special tokens
        UNK = self.UNK_TOKEN_IDX
        SOS = self.SOS_TOKEN_IDX
        EOS = self.EOS_TOKEN_IDX

        src_sentence = []
        tgt_sentence = []
        for word in raw_src_sentence:
            if word in src_word2idx: # src dictionary에 현재의 word가 있는 경우
                src_sentence.append(src_word2idx[word])
            else:
                src_sentence.append(UNK) # src dictionary에 현재의 word가 없는 경우 -> <unk> token
        
        for word in raw_tgt_sentence:
            if word in tgt_word2idx: # tgt dictionary에 현재의 word가 있는 경우
                tgt_sentence.append(tgt_word2idx[word])
            else:
                tgt_sentence.append(UNK) # tgt dictionary에 현재의 word가 없는 경우 -> <unk> token

        src_sentence = src_sentence[:max_len] # max_len까지의 sequence만
        tgt_sentence = [SOS] + tgt_sentence[:max_len-2] + [EOS] # SOS, EOS token을 추가하고 max_len까지의 sequence만

        return src_sentence, tgt_sentence

    def __getitem__(self, index: int) -> List[str]:
        return self._sentences[index]
    
    def __len__(self) -> int:
        return len(self._sentences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ko',
                        help='Type ko for korean or en for english')
    parser.add_argument('--tokenizer', type=str, default='mecab',
                        help='Input name of tokenizer that you want to use')
    args = parser.parse_args()

    tokenized = make_tokenized(args)
    print(tokenized)
