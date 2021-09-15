from konlpy import tag
from tqdm import tqdm
from transformers import BertTokenizer
import argparse
import spacy  # spacy version == 3.1.2, wrote in requirements.txt


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


class Language():
    def __init__(self) -> None:
        pass


def preprocess():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ko',
                        help='Type ko for korean or en for english')
    parser.add_argument('--tokenizer', type=str, default='mecab',
                        help='Input name of tokenizer that you want to use')
    args = parser.parse_args()

    tokenized = make_tokenized(args)
    print(tokenized)
