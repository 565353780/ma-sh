import os
import torch
from typing import Union

from ma_sh.Method.path import removeFile
from ma_sh.Module.refiner import Refiner

class RefineManager(object):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 10,
        sample_point_scale: float = 1.0,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warm_step_num: int = 40,
        factor: float = 0.8,
        patience: int = 2) -> None:
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.sample_point_scale = sample_point_scale
        self.use_inv = use_inv
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device
        self.lr = lr
        self.min_lr = min_lr
        self.warm_step_num = warm_step_num
        self.factor = factor
        self.patience = patience

        self.render = False
        self.render_freq = 1
        self.render_init_only = False
        self.save_result_folder_path = None
        self.save_log_folder_path = None
        return

    def createRefiner(self) -> Refiner:
        refiner = Refiner(
            self.anchor_num,
            self.mask_degree_max,
            self.sh_degree_max,
            self.mask_boundary_sample_num,
            self.sample_polar_num,
            self.sample_point_scale,
            self.use_inv,
            self.idx_dtype,
            self.dtype,
            self.device,
            self.lr,
            self.min_lr,
            self.warm_step_num,
            self.factor,
            self.patience,
            self.render,
            self.render_freq,
            self.render_init_only,
            self.save_result_folder_path,
            self.save_log_folder_path)

        return refiner

    def refineFile(
        self,
        mash_file_path: str,
        save_mash_file_path: str,
        save_pcd_file_path: Union[str, None]=None,
        overwrite: bool = False,
        print_progress: bool = False) -> bool:
        if not os.path.exists(mash_file_path):
            print('[ERROR][RefineManager::refineFile]')
            print('\t mash file not exist!')
            print('\t mash_file_path:', mash_file_path)

            return False

        if os.path.exists(save_mash_file_path):
            if not overwrite:
                return True

            removeFile(save_mash_file_path)

        refiner = self.createRefiner()

        if not refiner.loadParamsFile(mash_file_path):
            print('[ERROR][RefineManager::refineFile]')
            print('\t loadParamsFile failed!')
            return False

        refiner.autoTrainMash()
        refiner.mash.saveParamsFile(save_mash_file_path, overwrite)

        if save_pcd_file_path is not None:
            refiner.mash.saveAsPcdFile(save_pcd_file_path, overwrite, print_progress)

        return True

    def refineFolder(
        self,
        mash_folder_path: str,
        save_mash_folder_path: str,
        save_pcd_folder_path: Union[str, None]=None,
        overwrite: bool = False,
        print_progress: bool = False) -> bool:
        if not os.path.exists(mash_folder_path):
            print('[ERROR][RefineManager::refineFolder]')
            print('\t mash folder not exist!')
            print('\t mash_folder_path:', mash_folder_path)

            return False

        for root, _, files in os.walk(mash_folder_path):
            for file in files:
                if not file.endswith('.npy'):
                    continue

                rel_folder_path = os.path.relpath(root, mash_folder_path) + '/'
                mash_file_path = root + '/' + file
                save_mash_file_path = save_mash_folder_path + rel_folder_path + file
                save_pcd_file_path = None
                if save_pcd_folder_path is not None:
                    save_pcd_file_path = save_pcd_folder_path + rel_folder_path + file.replace('.npy', '.ply')

                if not self.refineFile(mash_file_path, save_mash_file_path, save_pcd_file_path, overwrite, print_progress):
                    continue

        return True
