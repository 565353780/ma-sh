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
    toMaskBoundaryMaxThetas,
    toInMaskSamplePolarIdxsList,
    toInMaskSamplePolarCounts,
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
    mask_boundary_base_values = toMaskBaseValues(mask_boundary_phis, mask_degree_max)
    mask_boundary_thetas = toMaskValues(mask_params, mask_boundary_base_values, mask_boundary_phi_idxs)
    return

def test():
    # setThread()

    sample_polar_num = 100
    anchor_num = 4
    mask_degree_max = 5
    mask_boundary_sample_num = 10
    idx_dtype = torch.int64
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
    mask_boundary_base_values = toMaskBaseValues(mask_boundary_phis, mask_degree_max)
    assert checkFormat(mask_boundary_base_values, dtype, device, [mask_degree_max * 2 + 1, anchor_num * mask_boundary_sample_num])

    mask_boundary_thetas = toMaskValues(mask_params, mask_boundary_base_values, mask_boundary_phi_idxs)
    assert checkFormat(mask_boundary_thetas, dtype, device, [anchor_num * mask_boundary_sample_num])

    mask_boundary_max_thetas = toMaskBoundaryMaxThetas(mask_boundary_thetas, mask_boundary_phi_idxs)
    assert checkFormat(mask_boundary_max_thetas, dtype, device, [anchor_num])

    #FIXME: may need to make dtype controllable in the future, now can only be torch.int64
    in_mask_sample_polar_idxs_list = toInMaskSamplePolarIdxsList(sample_thetas, mask_boundary_max_thetas)
    assert checkFormat(in_mask_sample_polar_idxs_list[0], idx_dtype, device)

    in_mask_sample_polar_counts = toInMaskSamplePolarCounts(in_mask_sample_polar_idxs_list)
    assert checkFormat(in_mask_sample_polar_counts, idx_dtype, device, [anchor_num])

    in_mask_sample_polar_idxs = toBoundIdxs(in_mask_sample_polar_counts)
    assert checkFormat(mask_boundary_phi_idxs, idx_dtype, device, [anchor_num + 1])
    print(in_mask_sample_polar_idxs)
    exit()

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
