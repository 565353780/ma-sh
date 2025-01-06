import torch
import numpy as np
from typing import Union
from torch.optim.adamw import AdamW

from ma_sh.Method.pcd import getPointCloud, downSample
from ma_sh.Model.mash import Mash
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Module.base_trainer import BaseTrainer

mode = 'mash'

class Refiner(BaseTrainer):
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
        warmup_step_num: int = 40,
        factor: float = 0.8,
        patience: int = 2,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = 1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.warmup_step_num = warmup_step_num
        self.factor = factor

        if mode == 'mash':
            self.mash = Mash(
                anchor_num,
                mask_degree_max,
                sh_degree_max,
                mask_boundary_sample_num,
                sample_polar_num,
                sample_point_scale,
                use_inv,
                idx_dtype,
                dtype,
                device,
            )
        elif mode == 'simple_mash':
            self.mash = SimpleMash(
                anchor_num,
                mask_degree_max,
                sh_degree_max,
                10,
                10,
                use_inv,
                idx_dtype,
                dtype,
                device,
            )

        self.optimizer = AdamW(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.rotate_vectors,
                self.mash.positions,
            ],
            lr=lr,
        )

        super().__init__(
            lr,
            min_lr,
            patience,
            render,
            render_freq,
            render_init_only,
            save_freq,
            save_result_folder_path,
            save_log_folder_path,
        )
        return

    def loadGTPoints(self, gt_points: np.ndarray, sample_point_num: Union[int, None] = None) -> bool:
        gt_pcd = getPointCloud(gt_points)

        sample_gt_pcd = gt_pcd
        if sample_point_num is not None:
            sample_gt_pcd = downSample(gt_pcd, sample_point_num)
            if sample_gt_pcd is None:
                print('[WARN][Trainer::loadGTPoints]')
                print('\t downSample failed! will use all input gt points!')
                sample_gt_pcd = gt_pcd

        gt_points = np.asarray(sample_gt_pcd.points)

        if self.mash.device == "cpu":
            gt_points_dtype = self.mash.dtype
        else:
            gt_points_dtype = torch.float32

        self.gt_points = (
            torch.from_numpy(gt_points)
            .type(gt_points_dtype)
            .to(self.mash.device)
            .reshape(1, -1, 3)
        )

        return True

    def loadParamsFile(self, mash_params_file_path: str) -> bool:
        if not self.mash.loadParamsFile(mash_params_file_path):
            print('[ERROR][Refiner::loadParamsFile]')
            print('\t loadParamsFile failed!')
            return False

        sample_pts = torch.vstack(self.mash.toSamplePoints()[:2]).cpu().numpy()

        self.loadGTPoints(sample_pts)
        return True

    def autoTrainMash(self,
                      gt_points_num: int = 400000,
                      ) -> bool:
        fit_loss_weight = 0.0
        coverage_loss_weight = 1.0
        boundary_connect_loss_weight = 1.0

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start warmUpEpoch...")

        self.warmUpEpoch(
            self.lr,
            self.gt_points,
            fit_loss_weight,
            coverage_loss_weight,
            boundary_connect_loss_weight,
            self.warmup_step_num
        )

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive lr...")
        current_lr = self.lr / self.factor

        while current_lr > self.min_lr:
            current_lr *= self.factor
            if current_lr < self.min_lr:
                current_lr = self.min_lr

            self.trainEpoch(
                current_lr,
                self.gt_points,
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            )

            if current_lr == self.min_lr:
                break

        self.autoSavePcd('final', add_idx=False)
        self.autoSaveMash('final')

        return True
