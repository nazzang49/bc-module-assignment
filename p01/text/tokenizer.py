from tqdm import tqdm
# 다양한 한국어 형태소 분석기가 클래스로 구현되어 있음
from konlpy import tag
from transformers import BertTokenizer
import spacy



def characterwise_tokenizer(sentence):
    syllables = []
    for word in sentence:
        for syllable in word:
            syllables.append(syllable)
        return syllables

tokenizers = {
    'okt': tag.Okt().morphs,  # 한국어 토크나이저
    #'mecab': tag.Mecab().morphs,  # 한국어 토크나이저
    'character': characterwise_tokenizer,  # 글자별 토크나이저
    #'spacy': spacy.load('en_core_web_sm'),
    'bert': BertTokenizer.from_pretrained('bert-base-cased').tokenize,
}


def make_tokenized(data, tokenizer = 'okt'):

    tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

    for sent in tqdm(data):
        tokens = tokenizers[tokenizer](sent)
        tokenized.append(tokens)

    return tokenized



if __name__ == '__main__':
    print('test')