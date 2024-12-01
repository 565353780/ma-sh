import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from copy import deepcopy
from torch.optim.adamw import AdamW

import mash_cpp

from ma_sh.Method.data import toNumpy
from ma_sh.Method.path import removeFile
from ma_sh.Method.pcd import getPointCloud, downSample
from ma_sh.Method.time import getCurrentTime
from ma_sh.Model.mash import Mash
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer

mode = 'mash'

class Refiner(object):
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
        patience: int = 2,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
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

        self.lr = lr
        self.min_lr = min_lr
        self.warm_step_num = warm_step_num
        self.factor = factor
        self.patience = patience

        self.epoch = 0
        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.gt_points = torch.tensor(0.0).type(dtype).to(device)

        self.optimizer = AdamW(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.rotate_vectors,
                self.mash.positions,
            ],
            lr=self.lr,
        )

        self.initRecords()

        self.o3d_viewer = None
        if self.render:
            self.o3d_viewer = O3DViewer()
            self.o3d_viewer.createWindow()

        self.min_lr_reach_time = 0

        # tmp
        self.translate = None
        self.scale = None
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

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

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray],
        sh_params: Union[torch.Tensor, np.ndarray],
        rotate_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        use_inv: bool,
    ) -> bool:
        self.mash.loadParams(mask_params, sh_params, rotate_vectors, positions, use_inv)
        return True

    def loadBestParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray],
        sh_params: Union[torch.Tensor, np.ndarray],
        rotate_vectors: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
        use_inv: bool,
    ) -> bool:
        if isinstance(mask_params, np.ndarray):
            mask_params = torch.from_numpy(mask_params)
        if isinstance(sh_params, np.ndarray):
            sh_params = torch.from_numpy(sh_params)
        if isinstance(rotate_vectors, np.ndarray):
            rotate_vectors = torch.from_numpy(rotate_vectors)
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions)

        self.best_params_dict["mask_params"] = mask_params
        self.best_params_dict["sh_params"] = sh_params
        self.best_params_dict["rotate_vectors"] = rotate_vectors
        self.best_params_dict["positions"] = positions
        self.best_params_dict["use_inv"] = use_inv
        return True

    def loadParamsFile(self, mash_params_file_path: str) -> bool:
        self.mash.loadParamsFile(mash_params_file_path)

        sample_pts = torch.vstack(self.mash.toSamplePoints()[:2]).cpu().numpy()

        self.loadGTPoints(sample_pts)
        return True

    def updateLr(self, lr: float) -> bool:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return True

    def trainStep(
        self,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
    ) -> Union[dict, None]:
        self.optimizer.zero_grad()

        boundary_pts, inner_pts = self.mash.toSamplePoints()[:2]

        fit_loss = torch.tensor(0.0).type(self.gt_points.dtype).to(self.gt_points.device)
        coverage_loss = torch.zeros_like(fit_loss)
        boundary_connect_loss = torch.zeros_like(fit_loss)

        if fit_loss_weight > 0 or coverage_loss_weight > 0:
            fit_loss, coverage_loss = mash_cpp.toChamferDistanceLoss(
                torch.vstack([boundary_pts, inner_pts]), self.gt_points
            )

        if boundary_connect_loss_weight > 0:
            boundary_connect_loss = mash_cpp.toBoundaryConnectLoss(
                self.mash.anchor_num, boundary_pts, self.mash.mask_boundary_phi_idxs
            )

        weighted_fit_loss = fit_loss_weight * fit_loss
        weighted_coverage_loss = coverage_loss_weight * coverage_loss
        weighted_boundary_connect_loss = (
            boundary_connect_loss_weight * boundary_connect_loss
        )

        loss = (
            weighted_fit_loss + weighted_coverage_loss + weighted_boundary_connect_loss
        )

        loss.backward()

        self.optimizer.step()

        loss_dict = {
            "State/boundary_pts": boundary_pts.shape[0],
            "State/inner_pts": inner_pts.shape[0],
            "Train/weighted_fit_loss": toNumpy(weighted_fit_loss),
            "Train/weighted_coverage_loss": toNumpy(weighted_coverage_loss),
            "Train/weighted_boundary_connect_loss": toNumpy(
                weighted_boundary_connect_loss
            ),
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": toNumpy(fit_loss) + toNumpy(coverage_loss),
        }

        return loss_dict

    def warmUpEpoch(
        self,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        self.mash.setGradState(True)

        print("[INFO][Refiner::warmUpEpoch]")
        print(
            "\t start train epoch :",
            self.epoch,
            "with lr : %.4f" % (self.lr * 1e3),
            "* 1e-3...",
        )
        pbar = tqdm(total=train_step_max)

        start_step = self.step

        for i in range(train_step_max):
            if self.render:
                if self.step % self.render_freq == 0:
                    self.renderMash()

            current_lr = self.lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)
            if self.logger.isValid():
                self.logger.addScalar("Train/lr", current_lr, self.step)

            loss_dict = self.trainStep(
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            )

            assert isinstance(loss_dict, dict)
            for key, item in loss_dict.items():
                self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSavePcd("train", 20)
            # self.autoSaveMash("train")

            self.step += 1
            pbar.update(1)

            self.logger.addScalar("Train/patience", self.patience, self.step)

            if start_step + self.step >= train_step_max:
                break

        self.logger.addScalar("Train/lr", self.lr, self.step - 1)

        self.epoch += 1
        return True

    def trainEpoch(
        self,
        epoch_lr: float,
        fit_loss_weight: float = 1.0,
        coverage_loss_weight: float = 1.0,
        boundary_connect_loss_weight: float = 1.0,
        train_step_max: Union[int, None] = None,
    ) -> bool:
        self.mash.setGradState(True)

        self.updateLr(epoch_lr)
        if self.logger.isValid():
            self.logger.addScalar("Train/lr", epoch_lr, self.step)

        print("[INFO][Refiner::trainEpoch]")
        print(
            "\t start train epoch :",
            self.epoch,
            "with lr : %.4f" % (epoch_lr * 1e3),
            "* 1e-3...",
        )
        if train_step_max is not None:
            pbar = tqdm(total=train_step_max)
        else:
            pbar = tqdm()

        start_step = self.step
        min_loss = float("inf")
        min_loss_reached_time = 0

        while True:
            if self.render:
                if self.step % self.render_freq == 0:
                    self.renderMash()

            loss_dict = self.trainStep(
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight
            )

            assert isinstance(loss_dict, dict)
            for key, item in loss_dict.items():
                self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSavePcd("train", 20)
            # self.autoSaveMash("train")

            self.step += 1
            pbar.update(1)

            if train_step_max is not None:
                self.logger.addScalar("Train/patience", self.patience, self.step)

                if start_step + self.step >= train_step_max:
                    break
            else:
                if loss < min_loss:
                    min_loss = loss
                    min_loss_reached_time = 0
                else:
                    min_loss_reached_time += 1

                self.logger.addScalar(
                    "Train/patience", self.patience - min_loss_reached_time, self.step
                )

                if min_loss_reached_time >= self.patience:
                    break

        self.logger.addScalar("Train/lr", epoch_lr, self.step - 1)

        self.epoch += 1
        return True

    def autoTrainMash(self) -> bool:
        fit_loss_weight = 0.0
        coverage_loss_weight = 1.0
        boundary_connect_loss_weight = 1.0

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start warmUpEpoch...")

        self.warmUpEpoch(
            fit_loss_weight,
            coverage_loss_weight,
            boundary_connect_loss_weight,
            self.warm_step_num
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
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            )

            if current_lr == self.min_lr:
                break

        self.autoSavePcd('final', add_idx=False)
        self.autoSaveMash('final')

        return True


    def toSamplePointsWithTransform(self) -> torch.Tensor:
        sample_mash = deepcopy(self.mash)

        if self.translate is not None:
            sample_mash.scale(self.scale, False)
            sample_mash.translate(self.translate)

        mask_boundary_sample_points, in_mask_sample_points = sample_mash.toSamplePoints()[:2]
        sample_points = torch.vstack([mask_boundary_sample_points, in_mask_sample_points])
        return sample_points

    def autoSaveMash(self, state_info: str, save_freq: int = 1, add_idx: bool = True) -> bool:
        if self.save_result_folder_path is None:
            return False

        if self.save_file_idx % save_freq != 0:
            if add_idx:
                self.save_file_idx += 1
            return False

        if self.save_file_idx not in [0, 20, 40, 60, 80, 100] and state_info != 'final':
            if add_idx:
                self.save_file_idx += 1
            return False

        save_file_path = (
            self.save_result_folder_path
            + 'mash/'
            + str(self.save_file_idx)
            + "_"
            + state_info
            + ".npy"
        )

        save_mash = deepcopy(self.mash)

        if self.translate is not None:
            save_mash.scale(self.scale, False)
            save_mash.translate(self.translate)

        save_mash.saveParamsFile(save_file_path, True)

        if add_idx:
            self.save_file_idx += 1
        return True

    def saveAsPcdFile(self, save_pcd_file_path: str, overwrite: bool = False) -> bool:
        if os.path.exists(save_pcd_file_path):
            if not overwrite:
                return True

            removeFile(save_pcd_file_path)

        save_mash = deepcopy(self.mash)

        if self.translate is not None:
            save_mash.scale(self.scale, False)
            save_mash.translate(self.translate)

        save_mash.saveAsPcdFile(save_pcd_file_path, overwrite)
        return True

    def autoSavePcd(self, state_info: str, save_freq: int = 1, add_idx: bool = True) -> bool:
        if self.save_result_folder_path is None:
            return False

        if self.save_file_idx % save_freq != 0:
            if add_idx:
                self.save_file_idx += 1
            return False

        if self.save_file_idx not in [0, 20, 40, 60, 80, 100] and state_info != 'final':
            if add_idx:
                self.save_file_idx += 1
            return False

        save_pcd_file_path = (
            self.save_result_folder_path
            + 'pcd/'
            + str(self.save_file_idx)
            + "_"
            + state_info
            + ".ply"
        )

        if not self.saveAsPcdFile(save_pcd_file_path, True):
            print('[ERROR][Trainer::autoSavePcd]')
            print('\t saveAsPcdFile failed!')
            return False

        if add_idx:
            self.save_file_idx += 1
        return True

    def renderMash(self) -> bool:
        if self.o3d_viewer is None:
            print("[ERROR][Trainer::renderMash]")
            print("\t o3d_viewer is None!")
            return False

        with torch.no_grad():
            self.o3d_viewer.clearGeometries()

            boundary_pts, inner_pts = self.mash.toSamplePoints()[:2]
            detect_points = torch.vstack([boundary_pts, inner_pts])
            detect_points = toNumpy(detect_points)
            pcd = getPointCloud(detect_points)
            self.o3d_viewer.addGeometry(pcd)

            """
            for j in range(self.mash.mask_params.shape[0]):
                view_cone = self.toO3DViewCone(j)
                view_cone.translate([-mesh_abb_length, 0, 0])
                self.o3d_viewer.addGeometry(view_cone)

                # inv_sphere = self.toO3DInvSphere(j)
                # inv_sphere.translate([-30, 0, 0])
                # self.o3d_viewer.addGeometry(inv_sphere)
            """

            self.o3d_viewer.update()

            if self.render_init_only:
                self.o3d_viewer.run()
                exit()
        return True
