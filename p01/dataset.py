from typing import List, Dict, Tuple, Sequence, Any
from preprocessing import preprocess, Language
from konlpy.tag import Okt
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch


class CBOWDataset(Dataset):
    """
    cbow_set = CBOWDataset(untokenized_Data)
    For convenience, I've put all tokenization and w2i build up process
    inside the Dataset Class. Thus, Tokenizing process is not needed.
    """

    def __init__(self, data, window_size=2):
        """
        Build CBOW Dataset, CBOW Dataset look at the words in
        window_size range and predict the center word.
        """
        self.x = []
        self.y = []
        self.train_tokenized = self.make_tokenized(data)
        self.w2i = self.make_w2i(self.train_tokenized)

        for tokens in self.train_tokenized:
            token_ids = [self.w2i[token] for token in tokens]
            for i, id in enumerate(token_ids):
                if i-window_size >= 0 and i+window_size < len(token_ids):
                    self.x.append(
                        token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
                    self.y.append(id)

        self.x = torch.LongTensor(self.x)  # (전체 데이터 개수, 2 * window_size)
        self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def make_tokenized(self, data):
        """
        Tokenize korean data based on Okt from Konlpy
        """
        tokenizer = Okt()
        tokenized = []
        for sent in data:
            tokens = tokenizer.morphs(sent, stem=True)
            tokenized.append(tokens)

        return tokenized

    def make_w2i(self, train_tokenized):
        """
        Construct w2i based on tokenized data
        """
        word_count = defaultdict(int)
        w2i = {}
        for tokens in train_tokenized:
            for token in tokens:
                word_count[token] += 1
        word_count = sorted(word_count.items(),
                            key=lambda x: x[1], reverse=True)
        for pair in word_count:
            if pair[0] not in w2i:
                w2i[pair[0]] = len(w2i)
        return w2i


class CBOWLoader(DataLoader):
    """
    just same as the DataLoader, but for convenience, inherited DataLoader
    and changed name as CBOWLoader

    Usage
    cbow_loader = CBOWLoader(cbow_set, batch_size=batch_size)
    It's equal to the command CBOWLoader name changed into Dataloader
    """

    def __init__(self, *args, **kwargs):
        super(CBOWLoader, self).__init__(*args, **kwargs)


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
            Arg:
                index : [int] 몇 번째 문장을 불러올지 지정
            return :
                src_sentence : List[int], tgt_sentence : List[int]
        """
        return preprocess(self._src[index], self._tgt[index], self._src.word2idx, self._tgt.word2idx, self._max_len)

    def __len__(self) -> int:
        return len(self._src)


class NMTDataLoder(DataLoader):
    def __init__(self,  *args, **kwargs):
        super(NMTDataLoder, self).__init__(*args, **kwargs)
