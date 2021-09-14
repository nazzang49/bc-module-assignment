from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch



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
    
    의사 코드
        1. 


    '''

    sorted_lens, sorted_idx = batch_lens.sort(descending=True)
    sorted_batch = batch[sorted_idx]

    embedding = nn.Embedding(vocab_size, embedding_size)
    sorted_batch_emb = embedding(sorted_batch)

    packed_batch = pack_padded_sequence(sorted_batch_emb.transpose(0, 1), sorted_lens)

    return packed_batch






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

    


