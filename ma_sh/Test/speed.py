import torch
from tqdm import trange

from ma_sh.Method.torch_thread import setThread
from ma_sh.Method.check import checkFormat
from ma_sh.Method.kernel import (
    toUniformSamplePhis,
    toUniformSampleThetas,
    toBoundIdxs,
    toMaskBoundaryPhis,
    toMaskBaseValues,
    toMaskValues,
)

def testPreLoadUniformSample(sample_polar_num, dtype=torch.float32, device='cpu'):
    sample_phis = toUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    sample_thetas = toUniformSampleThetas(sample_phis)
    return

def testPreLoadMaskBoundary(anchor_num, mask_boundary_sample_num, idx_dtype, dtype, device):
    mask_boundary_sample_nums = torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    mask_boundary_phi_idxs = toBoundIdxs(mask_boundary_sample_nums)
    mask_boundary_phis = toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num).type(dtype).to(device)
    return

def testMaskBoundary(mask_degree_max, mask_params, mask_boundary_phis, mask_boundary_phi_idxs):
    mask_boundary_base_values = toMaskBaseValues(mask_degree_max, mask_boundary_phis)
    mask_boundary_thetas = toMaskValues(mask_boundary_phi_idxs, mask_params, mask_boundary_base_values)
    return

def test():
    # setThread()

    sample_polar_num = 10000
    anchor_num = 4
    mask_degree_max = 5
    mask_boundary_sample_num = 10
    idx_dtype = torch.int16
    dtype = torch.float32
    device = 'cpu'

    # Params
    mask_params = torch.randn([anchor_num, mask_degree_max * 2 + 1]).type(dtype).to(device)
    assert checkFormat(mask_params, dtype, device, [anchor_num, mask_degree_max * 2 + 1])

    mask_params.requires_grad_(True)

    # Pre Load
    ## Uniform Sample
    sample_phis = toUniformSamplePhis(sample_polar_num).type(dtype).to(device)
    assert checkFormat(sample_phis, dtype, device, [sample_polar_num])

    sample_thetas = toUniformSampleThetas(sample_phis)
    assert checkFormat(sample_thetas, dtype, device, [sample_polar_num])

    ## Mask Boundary
    mask_boundary_sample_nums = torch.ones(anchor_num).type(idx_dtype).to(device) * mask_boundary_sample_num
    assert checkFormat(mask_boundary_sample_nums, idx_dtype, device, [anchor_num])

    mask_boundary_phi_idxs = toBoundIdxs(mask_boundary_sample_nums)
    assert checkFormat(mask_boundary_phi_idxs, idx_dtype, device, [anchor_num + 1])

    mask_boundary_phis = toMaskBoundaryPhis(anchor_num, mask_boundary_sample_num).type(dtype).to(device)
    assert checkFormat(mask_boundary_phis, dtype, device, [anchor_num * mask_boundary_sample_num])

    # Mask Boundary
    mask_boundary_base_values = toMaskBaseValues(mask_degree_max, mask_boundary_phis)
    assert checkFormat(mask_boundary_base_values, dtype, device, [mask_degree_max * 2 + 1, anchor_num * mask_boundary_sample_num])

    mask_boundary_thetas = toMaskValues(mask_boundary_phi_idxs, mask_params, mask_boundary_base_values)
    assert checkFormat(mask_boundary_thetas, dtype, device, [anchor_num * mask_boundary_sample_num])

    # Speed
    test_num = 1000

    print('[INFO][speed::test]')
    print('\t start speed test...')
    print('\t testPreLoadUniformSample')
    for _ in trange(test_num):
        testPreLoadUniformSample(sample_polar_num, dtype, device)

    print('\t testPreLoadMaskBoundary')
    for _ in trange(test_num):
        testPreLoadMaskBoundary(anchor_num, mask_boundary_sample_num, idx_dtype, dtype, device)

    print('\t testMaskBoundary')
    for _ in trange(test_num):
        testMaskBoundary(mask_degree_max, mask_params, mask_boundary_phis, mask_boundary_phi_idxs)
    return True
