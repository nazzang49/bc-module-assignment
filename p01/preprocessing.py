from konlpy import tag
from tqdm import tqdm
import spacy
from transformers import BertTokenizer
import argparse

def characterwise_tokenizer(sentence):
  syllables=[]
  for word in sentence:
    for syllable in word:
      syllables.append(syllable)
    return syllables

#Tokenizer dictionary
tokenizer = {
  'okt':tag.Okt().morphs,
  'mecab':tag.Mecab().morphs,
  'character':characterwise_tokenizer,
  'spacy':spacy.load('en_core_web_sm'),
  'bert':BertTokenizer.from_pretrained('bert-base-cased').tokenize,
}

#테스트용 데이터
data={'ko': [
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
'en':[
  'This assignment is about Natural Language Processing.',
],
'mixed':[
  'S를 원소가 하나인 집합은 닫힌 집합인 위상공간 X의 부분 집합이라고 하자.'
]
}

def make_tokenized(args):
  tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

  for sentence in tqdm(data[args.data]):
    tokens = tokenizer[args.tokenizer](sentence)
    tokenized.append(tokens)

  return tokenized

if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--data',type=str,default='ko',help='Type ko for korean or en for english')
  parser.add_argument('--tokenizer',type=str,default='mecab',help='Input name of tokenizer that you want to use')
  args=parser.parse_args()

  tokenized = make_tokenized(args)
  print(tokenized)

