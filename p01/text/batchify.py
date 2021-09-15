from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch

from typing import List, Dict, Tuple, Sequence, Any
from collections import Counter, defaultdict, OrderedDict
from itertools import chain
import random


random.seed(1234)


'''
batch 데이터를 만드는 함수에 대한 파일

배치화 방식
  1. 문장 패딩 처리 방식
    - padding_data, sort_pack_batch / bucketed_batch_indices

  2. 하나의 리스트인 전체 데이터를 정해진 batch 크기로 나누는 방식
    - batchify

한 번에 처리할 batch 불러오기
  - get_batch


collate_fn 만들기
  - collate_fn

'''


def padding_data (data, pad_id=0):
    '''
    Task : 토큰화된 데이터를 padding처리하여 batch 데이터로 만들기

    Task 설명 : data의 각 요소는 문장(sequence)을 의미하며 길이가 모두 다른 상태
            -> 가장 긴 문장(Maximum sequence)보다 길이가 작은 문장에 대해 
            0값(pad_id)을 채워서 가장 긴 문장의 길이로 바꾸는 작업 수행
            (이 때, 원래 문장의 길이에 대한 리스트 정의 : valid_lens)

    의사 코드
        1. 가장 긴문장 길이 구하기 (max_len)
        2. 각 문장에 대해
            2-1
            문장의 길이 리스트(valid_lens)에 문장 길이 추가

            2-2
            문장 = 문장 + [0] * (가장 긴문장 길이 - 문장 길이)
            으로 변환
        3. data를 Tensor로 변환
    '''

    # 1. 가장 긴문장 길이 구하기 (max_len)
    max_len = len(max(data, key=len))

    # 2. 각 문장에 대해
    valid_lens = []
    for i, seq in enumerate(tqdm(data)):
        # 2-1
        valid_lens.append(len(seq))

        # 2-2
        if len(seq) < max_len:
            data[i] = seq + [pad_id] * (max_len - len(seq))

    batch = torch.LongTensor(data)  # (B, L): (문장, padding된 문장 길이)
    batch_lens = torch.LongTensor(valid_lens)  # (B) : 원래 문장의 길이

    return batch, batch_lens




def sort_pack_batch(batch, batch_lens, embedding_size=256, vocab_size=100):

    '''
    Task : batch 데이터를 원본 데이터의 문장 길이순으로 내림차순 정렬 후 임베딩, pack 처리

    Task 설명 : batch 데이터에 padding 처리된 값은 실제 연산에 필요가 없다. 즉, 불피요한 연산이 된다.
               padding값이 연산에 들어가지 않도록 하는 작업을 해야한다.
            -> batch 데이터를 문장 길이(batch_lens) 순으로 1. 내림차순 정렬한 뒤 2. 임베딩하여 3. pack처리한다.
    '''
    # 1.
    # batch_len : batch의 각 요소에 대한 문장길이를 담은 배열
    # batch 정렬 - batch_len을 기준으로 내림차순 정렬
    sorted_lens, sorted_idx = batch_lens.sort(descending=True)
    sorted_batch = batch[sorted_idx]

    # 2.
    # vocab_size : 만들어진 dictionary의 길이(== 토큰화 된 값의 범위, 여기서는 100으로 설정)
    # embedding_size : 임베딩 이후 결과의 차원크기
    embedding = nn.Embedding(vocab_size, embedding_size)
    sorted_batch_emb = embedding(sorted_batch)

    # 3.
    # 정렬된 batch를 전치하여 입력, 정렬된 batch의 각 문장 길이 입력
    packed_batch = pack_padded_sequence(sorted_batch_emb.transpose(0, 1), sorted_lens)

    return packed_batch



def bucketed_batch_indices(
    sentence_length: List[Tuple[int, int]],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:


    '''
    Task : 문장의 길이에 따라 1.전체 데이터를 그룹화하여 2. 배치 단위로 만든 뒤  3.섞어주는 작업

    * Bucketing : 주어진 문장의 길이에 따라 데이터를 그룹화하여 padding을 적용하는 기법입니다. 
                  이 기법은 모델의 학습 시간을 단축하기 위해 고안되었습니다. 
                -> 손실 계산은 PAD 토큰을 고려하지 않지만, 실제로 GPU 리소스를 차지하고 성능을 저하시킵니다.

    아래 그림(필수과제4 그림 참조)과 같이 bucketing을 적용하지 않은 경우, 
    batch별 pad token의 개수가 늘어나 학습하는 데에 오랜 시간이 걸립니다. 


    예시 :
    sentence_length = [7, 4, 9, 2, 5, 10], batch_size = 3, and max_pad_len = 3
    가능한 batch_indices_list = [[0, 2, 5], [1, 3, 4]] 이다. -> [[7,9,10], [4,2,5]]의 값으로
    padding 처리시에 max_pad_len 값을 만족한다.
    

    input:
      sentence_length : (source_sentence 길이, target_sentence 길이)쌍에 대한 리스트
      * source_sentence : 번역 모델의 입력 문장, target_sentence: 번역 모델의 출력 문장
      batch_size : 한 번에 처리한 배치의 크기
      max_pad_len : padding처리시에 padding 토큰의 최대 갯수


    의사 코드

    1. 전체 데이터를 그룹화하기
        1-1. batch_map : (src, tgt) : [idx] 모양의 딕셔너리 정의
        1-2. min(source_sentence 길이, target_sentence 길이) 구하기
        1-3. 각 sentence_length[idx] : (source_sentence 길이, target_sentence 길이)에 대하여
             src, tgt 구하기 : (source_sentence 길이, target_sentence 길이) - min(source_sentence 길이, target_sentence 길이) + 1 를
             max_pad_len로 나눈 몫
             batch_map에 idx값 추가

    2. 배치 단위로 만들기
        2-1. batch_map의 각 value(idx) 리스트에 대해 정해놓은 배치크기 단위로 (인덱스) 리스트 만들기
        2-2. 만들어진 배치크기의 리스트를 batch_indices_list에 추가

    3. 배치 섞어주기


    return:
    batch_indices_list : batch화된 인덱스 리스트. 각 요소는 문장길이 목록의(==sentence_length에 대한) 인덱스 리스트이다.
    '''


    # 1.
    # 1-1.
    batch_map = defaultdict(list)
    batch_indices_list = []

    # 1-2.
    # 첫번째 인덱스인 src의 min length
    src_len_min = min(sentence_length, key=lambda item:item[0])[0] 
    # 두번째 인덱스인 tgt의 min length
    tgt_len_min = min(sentence_length, key=lambda item:item[1])[1] 

    # 1-3.
    for idx, (src_len, tgt_len) in enumerate(sentence_length):
        src = (src_len - src_len_min + 1) // max_pad_len # max_pad_len 단위로 묶어주기 위한 몫 (그림에서는 5)
        tgt = (tgt_len - tgt_len_min + 1) // max_pad_len # max_pad_len 단위로 묶어주기 위한 몫 (그림에서는 5)
        batch_map[(src, tgt)].append(idx)

    # 2.
    for key, value in batch_map.items():
        batch_indices_list += [value[i: i+batch_size] for i in range(0, len(value), batch_size)]

    # 3.
    # 배치의 길이로 인해 편향이 생길 수 있어 배치를 섞어주어야 한다.
    random.shuffle(batch_indices_list)

    return batch_indices_list


def batchify_bysize(data, bsz):

    '''
    Task : 하나의 리스트인 전체 데이터를 정해진 batch 크기(bsz)로 나누어 주기
          * 전체 데이터 : 모든 데이터를 하나의 리스트로 토큰화 한 데이터 (문장단위 X)
          * ex) 전체 데이터 크기 109, batch 크기 10  -> batch.shape : [10, 10], 9는 버린다.

    * batch작업의 결과로 실제 문장이 중간에 끊긴다면 서로의 연관성에 대해서는 학습되지 않는다.
    하지만 batch 처리에 있어 효육적이다. 
    
    '''

    # 디바이스 설정
    device = torch.device("cuda:0")

    # 배치 갯수 = 전체 데이터 크기 // batch 크기
    nbatch = data.size(0) // bsz

    # 데이터의 끝부분인 나머지를 없애는 작업
    data = data.narrow(0, 0, nbatch * bsz)

    # bsz 크기 단위로 (나머지가 없는) 전체 데이터를 나눕니다.
    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)



def get_batch(batch, i, bptt=35):
    
    '''
    Task : batch 데이터를 한 번에 처리할 갯수 만큼 반환해주기

    input
      batch : 배치 처리된 전체 데이터
      i : 가져올 데이터의 처음 인덱스 값 (i = bptt*n(번째 배치))
      btpp : 한 번에 처리할 배치 갯수
    
    seq_len : 나머지를 고려한 배치 갯수 (마지막 배치 갯수가 한번에 처리할 갯수보다 적을 때를 고려)

    '''

    seq_len = min(bptt, len(batch) - 1 - i)
    batch = batch[i:i+seq_len]
    target = batch[i+1:i+1+seq_len].view(-1)
    return batch, target


def collate_fn(
    batched_samples: List[Tuple[List[int], List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor]:

    '''
    Task
    각 문장의 길이가 가변적이므로 <PAD> 토큰을 사용하여 하나의 배치로 조합해야 합니다.
    즉, (소스/타겟)쌍의 문장 리스트를 소스문장 리스트, 타겟문장 리스트 배치로 조합하고 뒤에 <PAD> 토큰을 추가하는 collate_fn 기능 구현
    한편, 후자 구현의 편의를 위해 배치 내의 문장을 소스 문장 길이에 따라 내림차순으로 정렬해야 합니다.
    
    
    참고 1: 시계열 데이터의 전문가라면 [sequence_length, batch_size, ...]의 텐서가 [batch_size, sequence_length, ...]보다 훨씬 빠릅니다.
    다만, 직관적인 이해를 돕기 위해 이번에는 그냥 batch_first를 사용합니다.

    참고 2: collate_fn 인수를 사용하여 이 함수를 torch.utils.data.dataloader.DataLoader에 직접 적용할 수 있습니다.
    관심이 있는 경우 테스트 코드를 읽으십시오.

    힌트: torch.nn.utils.rnn.pad_sequence가 유용할 것입니다.

    Arguments:
    batched_samples -- (source_sentence, target_sentence)쌍을 요소로하는 리스트. 이 리스트는 배치로 변환되어야 합니다.


    의사 코드
    1. 하나의 배치(batched_samples)에 대하여 첫 번째 요소를 기준으로 길이에 따른 내림차순 정렬
    2. (source_sentence, target_sentence)쌍 리스트를 source_sentence, target_sentence로 변환
    3. source_sentence, target_sentence 0값으로 패딩 처리


    Return:
    src_sentences -- 배치처리된 source sentence
                        in shape (batch_size, max_src_sentence_length)
    tgt_sentences -- 배치처리된 target sentence
                        in shape (batch_size, max_tgt_sentence_length)
    '''


    # 1.
    batched_samples = sorted(batched_samples, key=lambda item:item[0], reverse= True) # 0번째 요소의 길이를 기준으로 내림차순 정렬
    
    # 2.
    src_sentences = []
    tgt_sentences = []
    for src_sentence, tgt_sentence in batched_samples:
        src_sentences.append(torch.tensor(src_sentence))
        tgt_sentences.append(torch.tensor(tgt_sentence))

    # 3.
    src_sentences = pad_sequence(src_sentences, batch_first=True) # batch x longest seuqence 순으로 정렬 (링크 참고)
    tgt_sentences = pad_sequence(tgt_sentences, batch_first=True) # batch x longest seuqence 순으로 정렬 (링크 참고)
    # 링크: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

    return src_sentences, tgt_sentences





if __name__ == '__main__':

    # 토큰화 처리된 data
    data = [
  [85,14,80,34,99,20,31,65,53,86,3,58,30,4,11,6,50,71,74,13],
  [62,76,79,66,32],
  [93,77,16,67,46,74,24,70],
  [19,83,88,22,57,40,75,82,4,46],
  [70,28,30,24,76,84,92,76,77,51,7,20,82,94,57],
  [58,13,40,61,88,18,92,89,8,14,61,67,49,59,45,12,47,5],
  [22,5,21,84,39,6,9,84,36,59,32,30,69,70,82,56,1],
  [94,21,79,24,3,86],
  [80,80,33,63,34,63],
  [87,32,79,65,2,96,43,80,85,20,41,52,95,50,35,96,24,80]
]

    # padding_data 함수 적용 결과
    # batch, batch_lens =  padding_data(data)
    # print(batch)
    # print(batch_lens)

    


