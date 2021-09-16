from text.tokenizer import make_tokenized
from text.makedictionary import make_word_count, make_word2index
import math
from collections import defaultdict
from tqdm import tqdm

import data


class NaiveBayesClassifier():

    '''
    Task : 특정 문서가 들어왔을 때 이를 적절한 클래스로 분류하는 모델
    * 클래스가 CV, NLP 두 개가 주어질 때 특정 문서가 들어오면 둘 중 적절한 클래스로 분류한다.
    * 이곳에서는 0,1 이진 분류의 경우만 구현되어있어 클래스가 3개 이상인 경우 추가 구현해야한다.


    Task 설명 : argmax of c 𝑃(c|d)를 구하는 것이 목표이고 이는 베이즈룰에 의해 다음과 같이 구해진다. 
               (𝑃(c|d) : 문서가 특정 클래스로 분류될 확률)
               = argmax of c 𝑃(𝑑|𝑐)*P(𝑐) = argmax of c 𝑃(𝑤1, 𝑤2, . . . , 𝑤n|𝑐)P(𝑐) → P(𝑐) ∏wi∈W 𝑃(wi|𝑐)
               (wi : 문서에 포함된 i번 째 단어)

    __init__
        self.k: Smoothing을 위한 상수.
        self.w2i: 사전에 구한 vocab.
        self.priors: 각 class의 prior 확률(: P(c)) -> classi : P(classi) 딕셔너리
        self.likelihoods: 각 token의 특정 class 조건 내에서의 likelihood. (: 𝑃(wi|𝑐)) 
                        -> tokeni : {class1: 𝑃(tokeni|𝑐lassi), ... , classN: 𝑃(tokenN|𝑐lassN)} 딕셔너리

    train
        1. set_priors : 사전확률(: P(c)) 구하기 = c 문서의 갯수 / 전체문서의 갯수
        2. set_likelihoods : 전체 문서에 등장하는 각 단어에 대한 likelihood(: 𝑃(wj|𝑐)) 구하기 
                            = c클래스에 등장하는 wj 갯수 / c클래스에 등장하는 전체 단어 갯수 

    inference
        tokens(token1, ... tokenN)가 주어질 때 argmax of c P(𝑐) ∏wi∈W 𝑃(wi|𝑐) 구하여 c값 반환
            
    '''

    def __init__(self, w2i, k=0.1):
        self.k = k
        self.w2i = w2i
        self.priors = {}
        self.likelihoods = {}

    def train(self, train_tokenized, train_labels):
        self.set_priors(train_labels)  # Priors 계산.
        self.set_likelihoods(train_tokenized, train_labels)  # Likelihoods 계산.

    def inference(self, tokens):
        log_prob0 = 0.0
        log_prob1 = 0.0

        for token in tokens:
            if token in self.likelihoods:  # 학습 당시 추가했던 단어에 대해서만 고려.
                log_prob0 += math.log(self.likelihoods[token][0])
                log_prob1 += math.log(self.likelihoods[token][1])

        # 마지막에 prior를 고려.
        log_prob0 += math.log(self.priors[0])
        log_prob1 += math.log(self.priors[1])

        if log_prob0 >= log_prob1:
            return 0
        else:
            return 1

    def set_priors(self, train_labels):
        class_counts = defaultdict(int)
        for label in tqdm(train_labels):
            class_counts[label] += 1

        for label, count in class_counts.items():
            self.priors[label] = class_counts[label] / len(train_labels)

    def set_likelihoods(self, train_tokenized, train_labels):
        token_dists = {}  # 각 단어의 특정 class 조건 하에서의 등장 횟수.
        class_counts = defaultdict(int)  # 특정 class에서 등장한 모든 단어의 등장 횟수.

        for i, label in enumerate(tqdm(train_labels)):
            count = 0
            for token in train_tokenized[i]:
                if token in self.w2i:  # 학습 데이터로 구축한 vocab에 있는 token만 고려.
                    if token not in token_dists:
                        token_dists[token] = {0:0, 1:0}
                        token_dists[token][label] += 1
                        count += 1
                class_counts[label] += count

        for token, dist in tqdm(token_dists.items()):
            if token not in self.likelihoods:
                self.likelihoods[token] = {
                    0:(token_dists[token][0] + self.k) / (class_counts[0] + len(self.w2i)*self.k),
                    1:(token_dists[token][1] + self.k) / (class_counts[1] + len(self.w2i)*self.k),
                }


if __name__ =='__main__':

    train_data = data.train_data
    train_labels = data.train_labels
    test_data = data.test_data

    # tokenize
    train_tokenized = make_tokenized(train_data)
    test_tokenized = make_tokenized(test_data)


    # print(train_tokenized)
    # print(test_tokenized)


    # 사전 만들기
    word_count = make_word_count(train_tokenized)
    w2i = make_word2index(word_count)

    # print(word_count)
    # print(w2i)


    # 모델 훈련하기
    classifier = NaiveBayesClassifier(w2i)
    classifier.train(train_tokenized, train_labels)


    # 모델 테스트하기
    preds = []
    for test_tokens in tqdm(test_tokenized):
        pred = classifier.inference(test_tokens)
        preds.append(pred)
    print(preds)