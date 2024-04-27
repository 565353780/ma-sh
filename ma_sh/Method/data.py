import torch
import numpy as np


def toNumpy(torch_data: torch.Tensor) -> np.ndarray:
    return torch_data.detach().clone().cpu().numpy()
