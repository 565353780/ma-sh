import os
import torch
from multiprocessing import cpu_count


def setThread(thread_num: int = cpu_count()):
    os.environ["OMP_NUM_THREADS"] = str(thread_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_num)
    os.environ["MKL_NUM_THREADS"] = str(thread_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_num)
    torch.set_num_threads(thread_num)
    return True
