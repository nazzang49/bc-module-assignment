from konlpy.tag import Okt
from tqdm import tqdm
from collections import defaultdict

def get_train_data():
    return [
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
    ]

def get_word_count(train_tokenized):
    '''
    Make counts of each words in tokenized train data

    :param train_tokenized: tokenized train data
    :return:
    '''
    word_count = defaultdict(int)
    for tokens in tqdm(train_tokenized):
        for token in tokens:
            word_count[token] += 1
    return word_count

def get_word_to_index(word_count):
    '''
    Make dictionary of 'word to index' by word count

    :param word_count: counts of each words in tokenized train data
    :return:
    '''
    w2i = {}
    for pair in tqdm(word_count):
        if pair[0] not in w2i:
            w2i[pair[0]] = len(w2i)
    return w2i

def make_tokenized(data):
    '''
    Make tokenized raw data based on morphs of Korean by Okt package

    :param data: train raw data
    :return:
    '''
    tokenizer = Okt()
    tokenized = []
    for sent in tqdm(data):
        tokens = tokenizer.morphs(sent, stem=True)
        tokenized.append(tokens)
    return tokenized

def check_train_data(train_data, train_tokenized, word_count, w2i):
    '''
    Check preprocessing procedure of train data

    :param train_data: raw data
    :param train_tokenized: tokenized raw data
    :param word_count: counts of each words in train_tokenized
    :param w2i: indices of each words in train_tokenized
    :return:
    '''
    print(' -- Check Train Raw Data --')
    print('==' * 30)
    print(train_data)

    print(' -- Check Train Tokenized --')
    print('==' * 30)
    print(train_tokenized)

    print(' -- Check Word Count of Train Tokenized --')
    print('==' * 30)
    print(word_count)

    print(' -- Check Word to Index of Word Count --')
    print('==' * 30)
    print(w2i)