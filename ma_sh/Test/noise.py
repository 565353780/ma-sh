import sys
sys.path.append('../distribution-manage/')

import torch
import numpy as np
import open3d as o3d
from ma_sh.Model.mash import Mash
from ma_sh.Config.transformer import getTransformer
from ma_sh.Method.io import loadMashFileParamsTensor

def test():
    mash_file_path = '/home/chli/chLi/Dataset/MashV4/ShapeNet/03001627/1006be65e7bc937e9141f9b58470d646.npy'
    transformer_id = 'ShapeNet_03001627'
    gauss_noise_level = 0.1
    render_dist = 3

    transformer = getTransformer(transformer_id)
    assert transformer is not None

    mash_pcd = Mash.fromParamsFile(mash_file_path).toSamplePcd()
    mash_pts = np.asarray(mash_pcd.points)
    mash_pts += np.random.randn(*mash_pts.shape) * gauss_noise_level
    mash_pcd.points = o3d.utility.Vector3dVector(mash_pts)

    mash_pcd.translate([-render_dist, 0, 0])
    mash_pcd.paint_uniform_color([1, 0, 0])

    mash_params = loadMashFileParamsTensor(mash_file_path)
    assert mash_params is not None
    mash_params += torch.randn_like(mash_params) * gauss_noise_level

    noise_mash = Mash()
    sh2d = 2 * 3 + 1
    ortho_poses = mash_params[:, :6]
    positions = mash_params[:, 6:9]
    mask_params = mash_params[:, 9 : 9 + sh2d]
    sh_params = mash_params[:, 9 + sh2d :]

    noise_mash.loadParams(
        mask_params=mask_params,
        sh_params=sh_params,
        positions=positions,
        ortho6d_poses=ortho_poses
    )

    noise_mash_pcd = noise_mash.toSamplePcd()

    noise_mash_pcd.paint_uniform_color([0, 0, 1])

    mash_params = loadMashFileParamsTensor(mash_file_path)
    assert mash_params is not None
    mash_params = transformer.transform(mash_params)
    mash_params += torch.randn_like(mash_params) * gauss_noise_level
    mash_params = transformer.inverse_transform(mash_params)

    noise_mash = Mash()
    sh2d = 2 * 3 + 1
    ortho_poses = mash_params[:, :6]
    positions = mash_params[:, 6:9]
    mask_params = mash_params[:, 9 : 9 + sh2d]
    sh_params = mash_params[:, 9 + sh2d :]

    noise_mash.loadParams(
        mask_params=mask_params,
        sh_params=sh_params,
        positions=positions,
        ortho6d_poses=ortho_poses
    )

    trans_noise_mash_pcd = noise_mash.toSamplePcd()

    trans_noise_mash_pcd.translate([render_dist, 0, 0])
    trans_noise_mash_pcd.paint_uniform_color([0, 1, 0])

    print('red: add noise to mash sampled points')
    print('blue: add noise to mash params')
    print('green: add noise to trans mash params')
    o3d.visualization.draw_geometries([mash_pcd, noise_mash_pcd, trans_noise_mash_pcd])

    return True
