from tqdm import tqdm
# 다양한 한국어 형태소 분석기가 클래스로 구현되어 있음
from konlpy import tag 


def make_tokenized(data):
    tokenizer = tag.Okt()
    
    tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

    for sent in tqdm(data):
        tokens = tokenizer.morphs(sent)
        tokenized.append(tokens)

    return tokenized



if __name__ == '__main__':
    print('test')