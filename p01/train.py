import torch
from importlib import import_module

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    입력된 tensor의 gradient 전파가 되지 않는 복사본을 반환합니다.
    """

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)