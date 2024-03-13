import torch


class MASH(object):
    def __init__(self, dtype, device: str) -> None:
        self.dtype = dtype
        self.device = device
        return
