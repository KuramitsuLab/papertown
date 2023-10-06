import torch

def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0