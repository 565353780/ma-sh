import torch
from tqdm import trange

from ma_sh.Method.torch_thread import setThread
from ma_sh.Method.kernel import (
    getUniformSamplePhis,
    getUniformSampleThetas,
    toBoundIdxs,
    getMaskBaseValues,
    getMaskValues,
)


def testMask(degree_max, params, phis, phi_idxs):
    base_values = getMaskBaseValues(degree_max, phis)
    mask_values = getMaskValues(phi_idxs, params, base_values)
    return mask_values

def testSamplePolars(sample_polar_num, dtype=torch.float32, device='cpu'):
    phis = getUniformSamplePhis(sample_polar_num, dtype, torch.device(device))
    thetas = getUniformSampleThetas(phis)
    return phis, thetas

def test():
    # setThread()

    sample_polar_num = 10000
    anchor_num = 40
    mask_degree_max = 5
    dtype = torch.float32
    device = torch.device('cpu')

    mask_params = torch.randn([anchor_num, mask_degree_max * 2 + 1]).type(dtype).to(device)
    assert mask_params.dtype == dtype
    assert mask_params.device == device

    phi_sample_nums = torch.randint(100, 1000, [anchor_num]).to(device)
    assert phi_sample_nums.dtype == torch.int64
    assert phi_sample_nums.device == device

    phi_idxs = toBoundIdxs(phi_sample_nums)
    assert phi_idxs.dtype == phi_sample_nums.dtype
    assert phi_idxs.device == device

    phis = torch.randn(phi_idxs[-1], requires_grad=True).type(dtype).to(device)
    assert phis.dtype == dtype
    assert phis.device == device
    assert phis.requires_grad

    # Uniform Sample
    sample_phis = getUniformSamplePhis(sample_polar_num, dtype, torch.device(device))
    assert sample_phis.dtype == dtype
    assert sample_phis.device == device

    sample_thetas = getUniformSampleThetas(sample_phis)
    assert sample_thetas.dtype == dtype
    assert sample_thetas.device == device

    # Mask
    base_values = getMaskBaseValues(mask_degree_max, phis)
    assert base_values.shape[0] == mask_params.shape[1]
    assert base_values.shape[1] == phi_idxs[-1].item()

    mask_values = getMaskValues(phi_idxs, mask_params, base_values)
    assert mask_values.shape[0] == phi_idxs[-1].item()

    # Speed
    for _ in trange(1000):
        testSamplePolars(sample_polar_num, dtype, device)

    for _ in trange(1000):
        testMask(mask_degree_max, mask_params, phis, phi_idxs)

    return True
