import torch
import numpy as np
import open3d as o3d
from typing import Union

from ma_sh.Config.mean_std import (
    POSITIONS_MEAN,
    POSITIONS_STD,
    ROTATE_VECTORS_MEAN,
    ROTATE_VECTORS_STD,
    MASK_PARAMS_MEAN,
    MASK_PARAMS_STD,
    SH_PARAMS_MEAN,
    SH_PARAMS_STD,
)
from ma_sh.Model.mash import Mash

def sampleRandomMashByRandn(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2,
    sample_num: int = 1,
    device: str = 'cpu',
    render: bool = False
) -> Mash:
    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    random_rorate_vectors = torch.randn((sample_num * anchor_num, 3))
    random_positions = torch.randn((sample_num * anchor_num, 3))
    random_mask_params = torch.randn((sample_num * anchor_num, mask_dim))
    random_sh_params = torch.randn((sample_num * anchor_num, sh_dim))

    if render:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 36, 1000, 0.8, True, torch.int64, torch.float64, device)
    else:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 0, 1, 1.0, True, torch.int64, torch.float64, device)

    mash.loadParams(
        mask_params=random_mask_params,
        sh_params=random_sh_params,
        positions=random_positions,
        rotate_vectors=random_rorate_vectors,
        use_inv=True,
    )

    return mash

def sampleRandomMashByRand(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2,
    sample_num: int = 1,
    device: str = 'cpu',
    render: bool = False
) -> Mash:
    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    std_weight = 10.0

    random_rorate_vectors = torch.hstack([(torch.rand((sample_num * anchor_num, 1)) - 0.5) * std_weight * ROTATE_VECTORS_STD[i] + ROTATE_VECTORS_MEAN[i] for i in range(3)])
    random_positions = torch.hstack([(torch.rand((sample_num * anchor_num, 1)) - 0.5) * std_weight * POSITIONS_STD[i] + POSITIONS_MEAN[i] for i in range(3)])
    random_mask_params = torch.hstack([(torch.rand((sample_num * anchor_num, 1)) - 0.5) * std_weight * MASK_PARAMS_STD[i] + MASK_PARAMS_MEAN[i] for i in range(mask_dim)])
    random_sh_params = torch.hstack([(torch.rand((sample_num * anchor_num, 1)) - 0.5) * std_weight * SH_PARAMS_STD[i] + SH_PARAMS_MEAN[i] for i in range(sh_dim)])

    if render:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 36, 1000, 0.8, True, torch.int64, torch.float64, device)
    else:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 0, 1, 1.0, True, torch.int64, torch.float64, device)

    mash.loadParams(
        mask_params=random_mask_params,
        sh_params=random_sh_params,
        positions=random_positions,
        rotate_vectors=random_rorate_vectors,
        use_inv=True,
    )

    return mash

def sampleRandomMashByNormal(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2,
    sample_num: int = 1,
    device: str = 'cpu',
    render: bool = False
) -> Mash:
    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    std_weight = 3.0

    random_rorate_vectors = torch.hstack([torch.normal(mean=ROTATE_VECTORS_MEAN[i], std=std_weight * ROTATE_VECTORS_STD[i], size=(sample_num * anchor_num, 1)) for i in range(3)])
    random_positions = torch.hstack([torch.normal(mean=POSITIONS_MEAN[i], std=std_weight * POSITIONS_STD[i], size=(sample_num * anchor_num, 1)) for i in range(3)])
    random_mask_params = torch.hstack([torch.normal(mean=MASK_PARAMS_MEAN[i], std=std_weight * MASK_PARAMS_STD[i], size=(sample_num * anchor_num, 1)) for i in range(mask_dim)])
    random_sh_params = torch.hstack([torch.normal(mean=SH_PARAMS_MEAN[i], std=std_weight * SH_PARAMS_STD[i], size=(sample_num * anchor_num, 1)) for i in range(sh_dim)])

    if render:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 36, 1000, 0.8, True, torch.int64, torch.float64, device)
    else:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 0, 1, 1.0, True, torch.int64, torch.float64, device)

    mash.loadParams(
        mask_params=random_mask_params,
        sh_params=random_sh_params,
        positions=random_positions,
        rotate_vectors=random_rorate_vectors,
        use_inv=True,
    )

    return mash
 
def sampleRandomMashBySphere(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2,
    sample_num: int = 1,
    device: str = 'cpu',
    render: bool = False
) -> Mash:
    mask_dim = 2 * mask_degree + 1
    sh_dim = (sh_degree + 1) ** 2

    random_face_forward_vectors = torch.randn((sample_num * anchor_num, 3))
    random_positions = torch.zeros((sample_num * anchor_num, 3))
    random_mask_params = torch.zeros((sample_num * anchor_num, mask_dim))
    random_sh_params = torch.zeros((sample_num * anchor_num, sh_dim))

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
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 36, 1000, 0.8, True, torch.int64, torch.float64, device)
    else:
        mash = Mash(sample_num * anchor_num, mask_degree, sh_degree, 0, 1, 1.0, True, torch.int64, torch.float64, device)

    mash.loadParams(
        mask_params=random_mask_params,
        sh_params=random_sh_params,
        positions=random_positions,
        use_inv=True,
        face_forward_vectors=random_face_forward_vectors,
    )

    return mash

def sampleRandomMashParams(
    anchor_num: int = 400,
    mask_degree: int = 3,
    sh_degree: int = 2,
    sample_num: int = 1,
    device: str = 'cpu',
    mode: str = 'randn',
    render: bool = False
) -> Union[torch.Tensor, None]:
    assert mode in ['randn', 'rand', 'normal', 'sphere']

    if mode == 'randn':
        mash = sampleRandomMashByRandn(
            anchor_num,
            mask_degree,
            sh_degree,
            sample_num,
            device,
            render)
    elif mode == 'rand':
        mash = sampleRandomMashByRand(
            anchor_num,
            mask_degree,
            sh_degree,
            sample_num,
            device,
            render)
    elif mode == 'normal':
        mash = sampleRandomMashByNormal(
            anchor_num,
            mask_degree,
            sh_degree,
            sample_num,
            device,
            render)
    elif mode == 'sphere':
        mash = sampleRandomMashBySphere(
            anchor_num,
            mask_degree,
            sh_degree,
            sample_num,
            device,
            render)
    else:
        print('[ERROR][random_mash::sampleRandomMashParams]')
        print('\t mode not valid!')
        return None

    if render:
        mash_pcd = mash.toSamplePcd()

        vertices = np.array([
            [-1.0, -1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0,  1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [-1.0,  1.0,  1.0]
        ])

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertices),
            lines=o3d.utility.Vector2iVector(lines)
        )

        o3d.visualization.draw_geometries([mash_pcd, line_set])

    random_mash_params = torch.cat((
        mash.toOrtho6DPoses().float(),
        mash.positions,
        mash.mask_params,
        mash.sh_params,
    ), dim=1).view(sample_num, anchor_num, -1)

    return random_mash_params
