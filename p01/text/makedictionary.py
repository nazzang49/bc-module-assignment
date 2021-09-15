from collections import defaultdict
from tqdm import tqdm


def make_word_count(data):

    '''
    input
    data - 하나의 문서 : [문장1, 문장2, ...]

    return
    word_count - token : token의 빈도수 로 이루어진 딕셔너리
    ex) {'안녕' : 55, '잘가' : 50 ...}
    '''
    word_count = defaultdict(int)

    for sentence in data:
        for token in sentence:
            word_count[token] += 1

    # token을 빈도수를 기준으로 내림차순 정렬
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    return word_count

def make_word2index(word_count):
    '''
    input
    word_count - token : token의 빈도수 로 이루어진 딕셔너리
    ex) {'안녕' : 55, '잘가' : 50 ...}

    return
    w2i - token : word_count에서 token의 index값 로 이루어진 딕셔너리
    ex) {'안녕' : 0, '잘가' : 1 ...}
    '''

    w2i = {}  # Key: 단어, Value: 단어의 index
    for pair in tqdm(word_count):
        if pair[0] not in w2i:
            w2i[pair[0]] = len(w2i)

    return w2i
