import torch
from tqdm import trange

from ma_sh.Method.torch_thread import setThread
from ma_sh.Method.check import checkFormat
from ma_sh.Method.kernel import (
    getUniformSamplePhis,
    getUniformSampleThetas,
    toBoundIdxs,
    getMaskBaseValues,
    getMaskValues,
)

def testSamplePolars(sample_polar_num, dtype=torch.float32, device='cpu'):
    phis = getUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    thetas = getUniformSampleThetas(phis)
    return phis, thetas

def testMask(degree_max, params, phis, phi_idxs):
    base_values = getMaskBaseValues(degree_max, phis)
    mask_values = getMaskValues(phi_idxs, params, base_values)
    return mask_values

def test():
    # setThread()

    sample_polar_num = 10000
    anchor_num = 40
    mask_degree_max = 5
    dtype = torch.float32
    device = 'cpu'

    mask_params = torch.randn([anchor_num, mask_degree_max * 2 + 1]).type(dtype).to(device)
    assert checkFormat(mask_params, dtype, device)

    phi_sample_nums = torch.randint(100, 1000, [anchor_num]).to(device)
    assert checkFormat(phi_sample_nums, torch.int64, device)

    phi_idxs = toBoundIdxs(phi_sample_nums)
    assert checkFormat(phi_idxs, phi_sample_nums.dtype, device)

    phis = torch.randn(phi_idxs[-1], requires_grad=True).type(dtype).to(device)
    assert checkFormat(phis, dtype, device)

    # Uniform Sample
    sample_phis = getUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    assert checkFormat(sample_phis, dtype, device, [sample_polar_num])

    sample_thetas = getUniformSampleThetas(sample_phis)
    assert checkFormat(sample_thetas, dtype, device, [sample_polar_num])

    # Mask
    base_values = getMaskBaseValues(mask_degree_max, phis)
    assert checkFormat(base_values, dtype, device, [mask_params.shape[1], phi_idxs[-1].item()])

    mask_values = getMaskValues(phi_idxs, mask_params, base_values)
    assert checkFormat(mask_values, dtype, device, [phi_idxs[-1].item()])

    # Speed
    for _ in trange(1000):
        testSamplePolars(sample_polar_num, dtype, device)

    for _ in trange(1000):
        testMask(mask_degree_max, mask_params, phis, phi_idxs)

    return True
