import os
import torch
import numpy as np
import open3d as o3d
from typing import Union

from ma_sh.Method.rotate import toOrthoPosesFromRotateVectors


def loadPcdFile(pcd_file_path):
    if not os.path.exists(pcd_file_path):
        print("[ERROR][mesh::loadPcdFile]")
        print("\t pcd file not exist!")
        print("\t pcd_file_path:", pcd_file_path)
        return None

    return o3d.io.read_point_cloud(pcd_file_path)


def loadMeshFile(mesh_file_path):
    if not os.path.exists(mesh_file_path):
        print("[ERROR][mesh::loadMeshFile]")
        print("\t mesh file not exist!")
        print("\t mesh_file_path:", mesh_file_path)
        return None

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh

def loadMashFileParamsTensor(
    mash_file_path: str,
    dtype = torch.float32,
    device: str = 'cpu') -> Union[torch.Tensor, None]:
    if not os.path.exists(mash_file_path):
        print('[ERROR][io::loadMashFileParamsTensor]')
        print('\t mash file not exist!')
        print('\t mash_file_path:', mash_file_path)
        return None

    mash_params = np.load(mash_file_path, allow_pickle=True).item()

    rotate_vectors = mash_params["rotate_vectors"]
    positions = mash_params["positions"]
    mask_params = mash_params["mask_params"]
    sh_params = mash_params["sh_params"]

    rotate_vectors_tensor = torch.tensor(rotate_vectors).to(device, dtype=dtype)
    positions_tensor = torch.tensor(positions).to(device, dtype=dtype)
    mask_params_tesnor = torch.tensor(mask_params).to(device, dtype=dtype)
    sh_params_tensor = torch.tensor(sh_params).to(device, dtype=dtype)

    ortho_poses_tensor = toOrthoPosesFromRotateVectors(rotate_vectors_tensor)

    mash_params = torch.cat((ortho_poses_tensor, positions_tensor, mask_params_tesnor, sh_params_tensor), dim=1)

    return mash_params
