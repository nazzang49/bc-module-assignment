from konlpy import tag
from tqdm import tqdm
import spacy
import re

def en_syllable_tokenizer(sentence):
    vowel={'a':1,'e':1,'i':1,'o':1,'u':1,'y':1}
    splitted_sentence=[]
    for word in sentence:
        syllables=[]
        pattern=re.compile('([a-z]h)|([a-z])')
        for character in re.split(pattern,word):
            if character:
                syllables.append(character)
        syllable=''

        for idx,alpha in enumerate(syllables):

            if syllables[idx-1] in vowel:
                if syllables[idx+1]

tokenizer = {'okt':tag.Okt().morphs,'mecab':tag.Mecab().morphs,'spacy':spacy.load('en_core_web_sm'),}

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


def make_tokenized(data):
  tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

  for sentence in tqdm(data):
    tokens = tokenizer['mecab'](sentence)
    tokenized.append(tokens)

  return tokenized

train_tokenized = make_tokenized(train_data)
print(train_tokenized)