import torch

from ma_sh.Config.weights import W0
from ma_sh.Model.mash import Mash

def sampleRandomMashParams(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2
) -> torch.Tensor:
    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    random_rotate_vectors = torch.randn((anchor_num, 3))
    random_positions = torch.zeros((anchor_num, 3))
    random_mask_params = torch.zeros((anchor_num, mask_dim))
    random_sh_params = torch.zeros((anchor_num, sh_dim))

    random_sh_params[:, 0] = 0.01 / W0[0]

    # mash = Mash(400, 3, 2, 0, 1, 1.0, True, torch.int64, torch.float64, 'cpu')
    mash = Mash(400, 3, 2, 36, 1000, 0.8, True, torch.int64, torch.float64, 'cuda')
    mash.loadParams(random_mask_params, random_sh_params, random_rotate_vectors, random_positions)
    mash.renderSamplePoints()
    exit()

    random_ortho_poses = mash.toOrtho6DPoses().float()

    random_mash_params = torch.cat((
        random_ortho_poses,
        random_positions,
        random_mask_params,
        random_sh_params
    ), dim=1)

    print(random_mash_params)
    print(random_mash_params.shape)
    exit()

    return random_mask_params
