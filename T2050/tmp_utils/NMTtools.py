import imp
from typing import List, Dict, Tuple, Sequence, Any
from collections import Counter, defaultdict
from itertools import chain
import random
import torch


class Language(Sequence[List[str]]):
    PAD_TOKEN = '<PAD>'
    PAD_TOKEN_IDX = 0
    UNK_TOKEN = '<UNK>'
    UNK_TOKEN_IDX = 1
    SOS_TOKEN = '<SOS>'
    SOS_TOKEN_IDX = 2
    EOS_TOKEN = '<EOS>'
    EOS_TOKEN_IDX = 3

    def __init__(self, sentences: List[str]) -> None:
        self._sentences: List[List[str]] = [sentence.split()
                                            for sentence in sentences]

        self.word2idx: Dict[str, int] = None
        self.idx2word: List[str] = None

    def build_vocab(self, min_freq: int = 1) -> None:
        SPECIAL_TOKENS: List[str] = [
            Language.PAD_TOKEN, Language.UNK_TOKEN, Language.SOS_TOKEN, Language.EOS_TOKEN]
        self.idx2word = SPECIAL_TOKENS + \
            [word for word, count in Counter(
                chain(*self._sentences)).items() if count >= min_freq]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

    def set_vocab(self, word2idx: Dict[str, int], idx2word: List[str]) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __getitem__(self, index: int) -> List[str]:
        return self._sentences[index]

    def __len__(self) -> int:
        return len(self._sentences)


def preprocess(
    raw_src_sentence: List[str],
    raw_tgt_sentence: List[str],
    src_word2idx: Dict[str, int],
    tgt_word2idx: Dict[str, int],
    max_len: int
) -> Tuple[List[int], List[int]]:
    """ Sentence preprocessor for neural machine translation

    Preprocess Rules:
    1. All words should be converted into their own index number by word2idx.
    2. If there is no matched word in word2idx, you should replace the word as <UNK> token.
    3. You have to use matched word2idx for each source/target language.
    4. You have to insert <SOS> as the first token of the target sentence.
    5. You have to insert <EOS> as the last token of the target sentence.
    6. The length of preprocessed sentences should not exceed max_len.
    7. If the lenght of the sentence exceed max_len, you must truncate the sentence.

    Arguments:
    raw_src_sentence -- raw source sentence without any modification
    raw_tgt_sentence -- raw target sentence without any modification 
    src_word2idx -- dictionary for source language which maps words to their unique numbers
    tgt_word2idx -- dictionary for target language which maps words to their unique numbers
    max_len -- maximum length of sentences

    Return:
    src_sentence -- preprocessed source sentence
    tgt_sentence -- preprocessed target sentence

    """
    # Special tokens, use these notations if you want
    UNK = Language.UNK_TOKEN_IDX
    SOS = Language.SOS_TOKEN_IDX
    EOS = Language.EOS_TOKEN_IDX

    # 아래에 코드 빈칸(None)을 완성해주세요
    # src_sentence = []
    # tgt_sentence = []
    # for word in raw_src_sentence:
    #     if word in src_word2idx: # src dictionary에 현재의 word가 있는 경우
    #         src_sentence.append(src_word2idx[word])
    #     else:
    #         src_sentence.append(UNK) # src dictionary에 현재의 word가 없는 경우

    # for word in raw_tgt_sentence:
    #     if word in tgt_word2idx: # tgt dictionary에 현재의 word가 있는 경우
    #         tgt_sentence.append(tgt_word2idx[word])
    #     else:
    #         tgt_sentence.append(UNK) # tgt dictionary에 현재의 word가 없는 경우

    # src_sentence = src_sentence[:max_len] # max_len까지의 sequence만
    # tgt_sentence = [SOS] + tgt_sentence[:max_len-2] + [EOS] # SOS, EOS token을 추가하고 max_len까지의 sequence만

    # # [선택] try, except을 활용해서 조금 더 빠르게 동작하는 코드를 작성해보세요.
    # for word in raw_src_sentence:
    #     try :
    #         src_sentence.append(src_word2idx[word])
    #     except :
    #         src_sentence.append(UNK)

    # for word in raw_tgt_sentence:
    #     try :
    #         tgt_sentence.append(tgt_word2idx[word])
    #     except :
    #         tgt_sentence.append(UNK)

    # src_sentence = src_sentence[:max_len] # max_len까지의 sequence만
    # tgt_sentence = [SOS] + tgt_sentence[:max_len-2] + [EOS] # SOS, EOS token을 추가하고 max_len까지의 sequence만

    # [선택] List Comprehension을 활용해서 짧은 코드를 작성해보세요. (~2 lines)
    src_sentence = []
    tgt_sentence = []
    for word in raw_src_sentence:
        src_sentence.append(
            src_word2idx[word]) if word in src_word2idx else src_sentence.append(UNK)

    for word in raw_tgt_sentence:
        tgt_sentence.append(
            tgt_word2idx[word]) if word in tgt_word2idx else tgt_sentence.append(UNK)

    src_sentence = src_sentence[:max_len]  # max_len까지의 sequence만
    tgt_sentence = [SOS] + tgt_sentence[:max_len-2] + \
        [EOS]  # SOS, EOS token을 추가하고 max_len까지의 sequence만
    # 코드 작성 완료
    return src_sentence, tgt_sentence


def bucketed_batch_indices(
    sentence_length: List[Tuple[int, int]],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:
    """ Function for bucketed batch indices
    Although the loss calculation does not consider PAD tokens,
    it actually takes up GPU resources and degrades performance.
    Therefore, the number of <PAD> tokens in a batch should be minimized in order to maximize GPU utilization.
    Implement a function which groups samples into batches that satisfy the number of needed <PAD> tokens in each sentence is less than or equals to max_pad_len.

    Note 1: several small batches which have less samples than batch_size are okay but should not be many. If you pass the test, it means "okay".

    Note 2: you can directly apply this function to torch.utils.data.dataloader.DataLoader with batch_sampler argument.
    Read the test codes if you are interested in.

    Arguments:
    sentence_length -- list of (length of source_sentence, length of target_sentence) pairs.
    batch_size -- batch size
    max_pad_len -- maximum padding length. The number of needed <PAD> tokens in each sentence should not exceed this number.

    return:
    batch_indices_list -- list of indices to be a batch. Each element should contain indices of sentence_length list.

    Example:
    If sentence_length = [7, 4, 9, 2, 5, 10], batch_size = 3, and max_pad_len = 3,
    then one of the possible batch_indices_list is [[0, 2, 5], [1, 3, 4]]
    because [0, 2, 5] indices has simialr length as sentence_length[0] = 7, sentence_length[2] = 9, and sentence_length[5] = 10.
    """

    # 아래에 코드 빈칸(None)을 완성해주세요
    batch_map = defaultdict(list)
    batch_indices_list = []

    src_len_min = min(sentence_length, key=lambda x: x[0])[
        0]  # 첫번째 인덱스인 src의 min length
    tgt_len_min = min(sentence_length, key=lambda x: x[1])[
        1]  # 두번째 인덱스인 tgt의 min length

    for idx, (src_len, tgt_len) in enumerate(sentence_length):
        # max_pad_len 단위로 묶어주기 위한 몫 (그림에서는 5)
        src = (src_len - src_len_min + 1) // (max_pad_len)
        # max_pad_len 단위로 묶어주기 위한 몫 (그림에서는 5)
        tgt = (tgt_len - tgt_len_min + 1) // (max_pad_len)
        batch_map[(src, tgt)].append(idx)

    for key, value in batch_map.items():
        batch_indices_list += [value[i: i+batch_size]
                               for i in range(0, len(value), batch_size)]

    # 코드 작성 완료

    # Don't forget shuffling batches because length of each batch could be biased
    random.shuffle(batch_indices_list)

    return batch_indices_list


def collate_fn(
    batched_samples: List[Tuple[List[int], List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Collate function
    Because each sentence has variable length, you should collate them into one batch with <PAD> tokens.
    Implement collate_fn function which collates source/target sentence into source/target batchs appending <PAD> tokens behind
    Meanwhile, for the convenience of latter implementations, you should sort the sentences within a batch by its source sentence length in descending manner.

    Note 1: if you are an expert on time-series data, you may know a tensor of [sequence_length, batch_size, ...] is much faster 
    than [batch_size, sequence_length, ...].
    However, for simple intuitive understanding, let's just use batch_first this time.

    Note 2: you can directly apply this function to torch.utils.data.dataloader.DataLoader with collate_fn argument.
    Read the test codes if you are interested in.

    Hint: torch.nn.utils.rnn.pad_sequence would be useful

    Arguments:
    batched_samples -- list of (source_sentence, target_sentence) pairs. This list should be converted to a batch

    Return:
    src_sentences -- batched source sentence
                        in shape (batch_size, max_src_sentence_length)
    tgt_sentences -- batched target sentence
                        in shape (batch_size, max_tgt_sentence_length)

    """
    PAD = Language.PAD_TOKEN_IDX
    batch_size = len(batched_samples)

    # 아래에 코드 빈칸을 완성해주세요
    # 0번째 요소의 길이를 기준으로 내림차순 정렬
    batched_samples = sorted(batched_samples, key=lambda x: x[0], reverse=True)

    src_sentences = []
    tgt_sentences = []
    for src_sentence, tgt_sentence in batched_samples:
        src_sentences.append(torch.tensor(src_sentence))
        tgt_sentences.append(torch.tensor(tgt_sentence))

    src_sentences = torch.nn.utils.rnn.pad_sequence(
        src_sentences, batch_first=True)  # batch x longest seuqence 순으로 정렬 (링크 참고)
    tgt_sentences = torch.nn.utils.rnn.pad_sequence(
        tgt_sentences, batch_first=True)  # batch x longest seuqence 순으로 정렬 (링크 참고)
    # 링크: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

    # 코드 작성 완료

    assert src_sentences.shape[0] == batch_size and tgt_sentences.shape[0] == batch_size
    assert src_sentences.dtype == torch.long and tgt_sentences.dtype == torch.long
    return src_sentences, tgt_sentences
