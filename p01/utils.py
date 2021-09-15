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
            w2i[pair[0]] = len(w2i) # length = idx
    return w2i

def make_tokenized(data):
    '''
    Make tokenized raw data based on morphs of Korean by Okt package

    :param data: train raw data
    :return:
    '''
    tokenizer = Okt()
    tokenized = []
    pos_list = []
    for sent in tqdm(data):
        # stem => if True, '맛있습니다' to '맛있다'
        # morphs from pos
        tokens = tokenizer.morphs(sent, stem=True)
        pos_list.append(tokenizer.pos(sent, stem=True))
        tokenized.append(tokens)
    print(' -- (Optional) POS of Tokens -- ')
    print('==' * 30)
    print(pos_list)
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

def make_skipgram_dataset(train_data):
    '''
    A whole procedure for making train data of skip-gram

    :param train_data: original train raw data
    :return:
    '''
    train_tokenized = make_tokenized(train_data)
    word_count = get_word_count(train_tokenized)
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return get_word_to_index(word_count)

def padding(data, is_src=True):
    '''
    Add pad to source and target data with (source or target max length - source or target length)

    :param data: source or target
    :param is_src: flag
    :return:
    '''
    max_len = len(max(data, key=len))
    print(f"Maximum Sequence Length: {max_len}")
    print(f"Is Source: {is_src}")

    valid_lens = []     # original sequence length
    pad_id = 0          # zero padding
    for i, seq in enumerate(tqdm(data)):
        valid_lens.append(len(seq))
        if len(seq) < max_len:
            # max 대비 남는 길이만큼 padding 처리
            data[i] = seq + [pad_id] * (max_len - len(seq))
    return data, valid_lens, max_len

def get_src_data():
    '''
    Return source data for Seq2Seq practice

    :return:
    '''
    return [
        [3, 77, 56, 26, 3, 55, 12, 36, 31],
        [58, 20, 65, 46, 26, 10, 76, 44],
        [58, 17, 8],
        [59],
        [29, 3, 52, 74, 73, 51, 39, 75, 19],
        [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93],
        [39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99, 5],
        [75, 34, 17, 3, 86, 88],
        [63, 39, 5, 35, 67, 56, 68, 89, 55, 66],
        [12, 40, 69, 39, 49]
    ]

def get_trg_data():
    '''
    Return target data for Seq2Seq practice

    :return:
    '''
    return [
        [75, 13, 22, 77, 89, 21, 13, 86, 95],
        [79, 14, 91, 41, 32, 79, 88, 34, 8, 68, 32, 77, 58, 7, 9, 87],
        [85, 8, 50, 30],
        [47, 30],
        [8, 85, 87, 77, 47, 21, 23, 98, 83, 4, 47, 97, 40, 43, 70, 8, 65, 71, 69, 88],
        [32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18],
        [37, 14, 49, 24, 93, 37, 54, 51, 39, 84],
        [16, 98, 68, 57, 55, 46, 66, 85, 18],
        [20, 70, 14, 6, 58, 90, 30, 17, 91, 18, 90],
        [37, 93, 98, 13, 45, 28, 89, 72, 70]
    ]

def check_seq2seq_data(src_data, trg_data):
    '''
    Check Seq2Seq train data

    :param src_data:
    :param trg_data:
    :return:
    '''
    print(' -- Check Source Data After Padding --')
    print('==' * 30)
    print(src_data)

    print(' -- Check Target Data After Padding --')
    print('==' * 30)
    print(trg_data)

def make_seq2seq_dataset():
    '''
    A whole procedure for making train data of Seq2Seq

    :return: dictionaries of src and trg
    '''
    src_data = get_src_data()
    trg_data = get_trg_data()
    src_data, src_lens, src_max_len = padding(src_data)
    trg_data, trg_lens, trg_max_len = padding(trg_data)
    check_seq2seq_data(src_data, trg_data)

    src_dict = {
        'src_data': src_data,
        'src_lens': src_lens,
        'src_max_len': src_max_len,
    }

    trg_dict = {
        'trg_data': trg_data,
        'trg_lens': trg_lens,
        'trg_max_len': trg_max_len,
    }
    return src_dict, trg_dict