import torch

from ma_sh.Config.mean_std import (
    POSITIONS_MEAN,
    POSITIONS_STD,
    MASK_PARAMS_MEAN,
    SH_PARAMS_MEAN,
)
from ma_sh.Model.mash import Mash

def sampleRandomMashParams(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2,
    render: bool = False
) -> torch.Tensor:
    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    random_face_forward_vectors = torch.randn((anchor_num, 3))
    random_positions = torch.zeros((anchor_num, 3))
    random_mask_params = torch.zeros((anchor_num, mask_dim))
    random_sh_params = torch.zeros((anchor_num, sh_dim))

    random_face_forward_vectors_norm = torch.norm(random_face_forward_vectors, dim=1)
    random_face_forward_vectors_norm[random_face_forward_vectors_norm == 0.0] = 1.0
    random_face_forward_directions = random_face_forward_vectors / random_face_forward_vectors_norm.unsqueeze(1)

    for i in range(3):
        random_positions[:, i] = POSITIONS_MEAN[i] - POSITIONS_STD[i] * random_face_forward_directions[:, i]

    for i in range(mask_dim):
        random_mask_params[:, i] = MASK_PARAMS_MEAN[i]
    for i in range(sh_dim):
        random_sh_params[:, i] = SH_PARAMS_MEAN[i]

    if render:
        mash = Mash(400, 3, 2, 36, 1000, 0.8, True, torch.int64, torch.float64, 'cuda')
    else:
        mash = Mash(400, 3, 2, 0, 1, 1.0, True, torch.int64, torch.float64, 'cpu')

    mash.loadParams(
        mask_params=random_mask_params,
        sh_params=random_sh_params,
        positions=random_positions,
        use_inv=True,
        face_forward_vectors=random_face_forward_vectors,
    )

    if render:
        mash.renderSamplePoints()


    random_mash_params = torch.cat((
        mash.toOrtho6DPoses().float(),
        mash.positions,
        mash.mask_params,
        mash.sh_params
    ), dim=1)

    return random_mash_params
