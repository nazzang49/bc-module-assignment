from text.tokenizer import make_tokenized
from text.makedictionary import make_word_count, make_word2index
import math
from collections import defaultdict
from tqdm import tqdm


class NaiveBayesClassifier():

    '''
    self.k: Smoothing을 위한 상수.
    self.w2i: 사전에 구한 vocab.
    self.priors: 각 class의 prior 확률.
    self.likelihoods: 각 token의 특정 class 조건 내에서의 likelihood.
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

    train_data = [
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
    train_labels = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]

    test_data = [
    "정말 좋았습니다. 또 가고 싶네요.",
    "별로였습니다. 되도록 가지 마세요.",
    "다른 분들께도 추천드릴 수 있을 만큼 만족했습니다.",
    "서비스가 좀 더 개선되었으면 좋겠습니다. 기분이 좀 나빴습니다."
    ]

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