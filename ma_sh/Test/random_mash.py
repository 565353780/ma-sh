from ma_sh.Method.random_mash import sampleRandomMashParams

def test():
    anchor_num = 400
    mask_degree = 3
    sh_degree = 2
    sample_num = 1
    device = 'cuda'
    mode = 'normal'
    render = True

    random_mash_params = sampleRandomMashParams(anchor_num, mask_degree, sh_degree, sample_num, device, mode, render)
    assert random_mash_params is not None

    print('random_mash_params.shape: ')
    print(random_mash_params.shape)

    return True
