from typing import List, Dict, Tuple, Sequence, Any
import torch
from collections import Counter
from itertools import chain
from NMTDataset import NMTDataset

"""
    해당 모듈에서 빠져 있는 함수, 모듈들
        1. collate_fn : 창한님이 완성하신 모듈에서 import 할 것!
        2. bucketed_batch_indices : 창한님이 완성하신 모듈에서 import 할 것!
        3. Language : 혜수님이 완성하신 모듈에서 import 할 것 !
"""


def test_dataloader():
    print("======Dataloader Test======")
    english = Language(     # Language 객체로 enlish 선언
        ['고기를 잡으러 바다로 갈까나', '고기를 잡으러 강으로 갈까나', '이 병에 가득히 넣어가지고요'])
    korean = Language(['Shall we go to the sea to catch fish ?',
                      'Shall we go to the river to catch fish', 'Put it in this bottle'])
    english.build_vocab()   # word2idx, idx2word 생성
    korean.build_vocab()    # word2idx, idx2word 생성
    dataset = NMTDataset(src=english, tgt=korean)   # dataset 생성

    batch_size = 4
    max_pad_len = 5
    sentence_length = list(
        map(lambda pair: (len(pair[0]), len(pair[1])), dataset))
    """
        dataset의 index 접근으로 인해 idx번째 문장의 src, tgt을 word2idx로 인코딩 된 값들의 길이가 tuple 형태로 짝을 이뤄 list에 저장된다.

        pair[0] = src_sentence : List[int] / pair[1] = tgt_sentence : List[int]
    """

    bucketed_batch_indices(
        sentence_length, batch_size=batch_size, max_pad_len=max_pad_len)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, collate_fn=collate_fn, num_workers=2,
                                                        batch_sampler=bucketed_batch_indices(sentence_length, batch_size=batch_size, max_pad_len=max_pad_len))
    src_sentences, tgt_sentences = next(iter(dataloader))
    print("Tensor for Source Sentences: \n", src_sentences)
    print("Tensor for Target Sentences: \n", tgt_sentences)

    print("모든 전처리 과제를 완료했습니다 고생하셨습니다 :)")


test_dataloader()
