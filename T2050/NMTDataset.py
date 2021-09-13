
from typing import List, Dict, Tuple, Sequence, Any, Language

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
