import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union, Tuple
from multiprocessing import Pool

from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Method.rotate import toOrthoPosesFromRotateVectors


def saveLargeNPY(data: np.ndarray, save_npy_file_path: str, chunk_size: int=1000, overwrite: bool=False) -> bool:
    if not overwrite:
        if os.path.exists(save_npy_file_path):
            return True

        removeFile(save_npy_file_path)

    total_size = data.shape[0]
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    createFileFolder(save_npy_file_path)

    tmp_save_npy_file_path = save_npy_file_path[:-4] + '_tmp.npy'

    print('[INFO][io::saveLargeNPY]')
    print('\t start save npy file:', save_npy_file_path)
    with open(tmp_save_npy_file_path, 'wb') as f:
        np.save(f, np.empty((0,) + data.shape[1:], dtype=data.dtype))
        for i in tqdm(range(num_chunks), desc="Saving progress"):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_size)
            chunk = data[start:end]
            np.lib.format.write_array(f, chunk, allow_pickle=False)

    renameFile(tmp_save_npy_file_path, save_npy_file_path)

    return True

def loadLargeNPY(npy_file_path: str, chunk_size: int=1000) -> Union[np.ndarray, None]:
    if not os.path.exists(npy_file_path):
        print('[ERROR][io::loadLargeNPY]')
        print('\t npy file not exist!')
        return None

    print('[INFO][io::loadLargeNPY]')
    print('\t start load npy file:', npy_file_path)
    with open(npy_file_path, 'rb') as f:
        header = np.lib.format.read_array_header_1_0(f)
        dtype = header[2]
        shape = header[0]
        total_size = shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        array = np.empty(shape, dtype=dtype)

        for i in tqdm(range(num_chunks), desc="Loading progress"):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_size)
            array[start:end] = np.lib.format.read_array(f)

    return array

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

    rotate_vectors_tensor = torch.tensor(rotate_vectors).to(device, dtype=torch.float64)
    positions_tensor = torch.tensor(positions).to(device, dtype=torch.float64)
    mask_params_tesnor = torch.tensor(mask_params).to(device, dtype=torch.float64)
    sh_params_tensor = torch.tensor(sh_params).to(device, dtype=torch.float64)

    ortho_poses_tensor = toOrthoPosesFromRotateVectors(rotate_vectors_tensor)

    mash_params = torch.cat((ortho_poses_tensor, positions_tensor, mask_params_tesnor, sh_params_tensor), dim=1)

    mash_params = mash_params.to(dtype)

    return mash_params

def loadMashFile(mash_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params_dict = np.load(mash_file_path, allow_pickle=True).item()
    rotate_vectors = params_dict["rotate_vectors"]
    positions = params_dict["positions"]
    mask_params = params_dict["mask_params"]
    sh_params = params_dict["sh_params"]

    return rotate_vectors, positions, mask_params, sh_params

def loadMashFolder(mash_folder_path: str,
                   keep_dim: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[None, None, None, None]]:
    if not os.path.exists(mash_folder_path):
        print('[ERROR][io::loadMashFolder]')
        print('\t mash folder not exist!')
        print('\t mash_folder_path:', mash_folder_path)

        return None, None, None, None

    mash_file_path_list = []
    for root, _, files in os.walk(mash_folder_path):
        for file in files:
            if not file.endswith('.npy'):
                continue

            mash_file_path_list.append(root + '/' + file)

    mash_file_path_list.sort()

    print('[INFO][io::loadMashFolder]')
    print('\t start load mash files...')
    with Pool(os.cpu_count()) as pool:
        result_list = list(tqdm(pool.imap(loadMashFile, mash_file_path_list), total=len(mash_file_path_list)))

    if keep_dim:
        rotate_vectors_array = np.stack([result[0] for result in result_list], axis=0)
        positions_array = np.stack([result[1] for result in result_list], axis=0)
        mask_params_array = np.stack([result[2] for result in result_list], axis=0)
        sh_params_array = np.stack([result[3] for result in result_list], axis=0)
    else:
        rotate_vectors_array = np.vstack([result[0] for result in result_list])
        positions_array = np.vstack([result[1] for result in result_list])
        mask_params_array = np.vstack([result[2] for result in result_list])
        sh_params_array = np.vstack([result[3] for result in result_list])

    return rotate_vectors_array, positions_array, mask_params_array, sh_params_array
