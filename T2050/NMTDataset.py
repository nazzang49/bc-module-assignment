from typing import List, Dict, Tuple, Sequence, Any
from tmp_utils.NMTtools import Language
import argparse
"""
    해당 모듈에서 빠져 있는 함수    
        1. preprocess
        : 혜수님이 완성 시킨 것 preprocess을 import 해야함 !!~~
"""


class NMTDataset(Sequence[Tuple[List[int], List[int]]]):
    def __init__(self, src: Language, tgt: Language, max_len: int = 30) -> None:
        """
            Language 클래스로 생성된 객체 src, tgt
            src : 번역을 할 문장들
            tgt : 번역을 된 문장들
        """
        assert len(src) == len(tgt)  # 번역 해야하는 문장의 개수와 번역 된 문장의 개수가 같아야한다.
        # Language에서 bulid_vocab 이나 set_vocab으로 word2idx가 존재해야한다.
        assert src.word2idx is not None and tgt.word2idx is not None

        self._src = src
        self._tgt = tgt
        self._max_len = max_len

    def __getitem__(self, index: int) -> Tuple[List[str], List[str]]:
        """
            NMTDataset에 index로 접근( NMTDataset[index] )할 경우 해당 문장을 지정한 maxlen까지의 길이를 가진 word2idx로 변화한 sentance 값을 return

            return :
                src_sentence : List[int], tgt_sentence : List[int]
        """
        return preprocess(self._src[index], self._tgt[index], self._src.word2idx, self._tgt.word2idx, self._max_len)

    def __len__(self) -> int:
        return len(self._src)


if __name__ == '__main__':
    from tmp_utils.NMTtools import *
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", help="number of batch size (default : 3)", default=3, type=int)
    parser.add_argument(
        "--max_pad_len", help="length of max pad (default : 5)", default=5, type=int)
    arg = parser.parse_args()
    batch_size = arg.batch_size
    max_pad_len = arg.max_pad_len

    print("======Dataloader Test======")
    english = Language(
        ['고기를 잡으러 바다로 갈까나', '고기를 잡으러 강으로 갈까나', '이 병에 가득히 넣어가지고요'])
    korean = Language(['Shall we go to the sea to catch fish ?',
                      'Shall we go to the river to catch fish', 'Put it in this bottle'])
    english.build_vocab()
    korean.build_vocab()
    dataset = NMTDataset(src=english, tgt=korean)
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
