from ma_sh.Method.random_mash import sampleRandomMashParams

def test():
    anchor_num = 400
    mask_degree = 3
    sh_degree = 2

    random_mash_params = sampleRandomMashParams(anchor_num, mask_degree, sh_degree)

    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    random_ortho_poses = random_mash_params[:, :6]
    random_positions = random_mash_params[:, 6: 9]

    return True
