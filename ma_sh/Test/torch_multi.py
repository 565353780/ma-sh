import torch
from tqdm import trange

from ma_sh.Method.idx import toBoundIdxs
from ma_sh.Method.mask import getSH2DBaseValues, getSH2DValues

import os
from multiprocessing import cpu_count

cpu_num = cpu_count() # 自动获取最大核心数目
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def merge(sh2d_degree, params, phis, phi_idxs):
    base_values = getSH2DBaseValues(sh2d_degree, phis)
    sh2d_values = getSH2DValues(phi_idxs, params, base_values)
    return True

def test():
    anchor_num = 40
    sh2d_degree = 5
    #FIXME: why cpu is more faster than gpu?
    device = 'cpu'

    params = torch.randn([anchor_num, sh2d_degree * 2 + 1]).type(torch.float32).to(device)

    phi_sample_nums = torch.tensor([1000, 500, 800, 400]).type(torch.int).to(device)

    phi_idxs = toBoundIdxs(phi_sample_nums)

    phis = torch.randn(phi_idxs[-1], requires_grad=True).type(torch.float32).to(device)

    merge(sh2d_degree, params, phis, phi_idxs)

    for _ in trange(10000):
        merge(sh2d_degree, params, phis, phi_idxs)

    return True
