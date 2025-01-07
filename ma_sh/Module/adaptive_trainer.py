import torch
import numpy as np
import open3d as o3d
from math import sqrt
from typing import Union
from torch.optim import AdamW

from ma_sh.Config.weights import W0
from ma_sh.Config.constant import EPSILON
from ma_sh.Method.pcd import downSample
import mash_cpp

from ma_sh.Model.mash import Mash
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.data import toNumpy
from ma_sh.Module.base_trainer import BaseTrainer

mode = 'mash'

class AdaptiveTrainer(BaseTrainer):
    def __init__(
        self,
        init_anchor_num: int = 40,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cuda",
        max_fit_error: float = 1e-3,
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        warmup_epoch: int = 4,
        factor: float = 0.8,
        patience: int = 2,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = 1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.max_fit_error = max_fit_error
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor

        self.max_fit_error2 = self.max_fit_error * self.max_fit_error

        self.fit_loss_weight = 1.0
        self.fit_loss_scale = 1.1

        if mode == 'mash':
            self.mash = Mash(
                init_anchor_num,
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
                init_anchor_num,
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

        self.add_anchor_num = 20
        self.add_anchor_freq = 20
        self.coverage_percent = 0
        return

    def updateOptimizer(self, lr: float) -> bool:
        self.optimizer = AdamW(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.rotate_vectors,
                self.mash.positions,
            ],
            lr=lr,
        )
        return True

    @torch.no_grad()
    def getCoveragePercent(self, coverage_dists2: torch.Tensor) -> float:
        coveraged_num = (coverage_dists2 < 2.0 * self.max_fit_error2).sum().item()
        total_elements = coverage_dists2.numel()
        coverage_percent = 100.0 * coveraged_num / total_elements
        return coverage_percent

    @torch.no_grad()
    def addAnchor(self, coverage_dists2: torch.Tensor) -> bool:
        if (self.step + 1) % self.add_anchor_freq != 0:
            return True
        return True

        surface_dist = 0.001

        not_fitted_point_idxs = coverage_dists2.detach().clone().cpu().flatten() > 2.0 * self.max_fit_error2

        not_fitted_points = self.gt_points[not_fitted_point_idxs]
        not_fitted_normals = self.gt_normals[not_fitted_point_idxs]

        not_fitted_pcd = o3d.geometry.PointCloud()
        not_fitted_pcd.points = o3d.utility.Vector3dVector(not_fitted_points)
        not_fitted_pcd.normals = o3d.utility.Vector3dVector(not_fitted_normals)

        anchor_pcd = downSample(not_fitted_pcd, self.add_anchor_num)

        if anchor_pcd is None:
            print("[ERROR][AdaptiveTrainer::addAnchor]")
            print("\t downSample failed!")
            return False

        sample_pts = np.asarray(anchor_pcd.points)
        sample_normals = np.asarray(anchor_pcd.normals)

        sample_pts = torch.from_numpy(sample_pts).to(self.mash.device, dtype=self.mash.dtype)
        sample_normals = torch.from_numpy(sample_normals).to(self.mash.device, dtype=self.mash.dtype)

        self.mash.setGradState(False)

        self.mash.updateAnchorNum(self.mash.anchor_num + self.add_anchor_num)

        new_sh_params = torch.ones([self.add_anchor_num, self.mash.sh_params.shape[1]],
                                   dtype=self.mash.dtype, device=self.mash.device) * EPSILON
        new_sh_params[:, 0] = surface_dist / W0[0]
        self.mash.sh_params.data[-self.add_anchor_num:, :] = new_sh_params

        new_positions = sample_pts + surface_dist * sample_normals
        self.mash.positions.data[-self.add_anchor_num:, :] = new_positions

        new_rotate_vectors = mash_cpp.toRotateVectorsByFaceForwardVectors(
            -sample_normals
        )
        self.mash.rotate_vectors.data[-self.add_anchor_num:, :] = new_rotate_vectors

        self.mash.regularRotateVectors()

        self.mash.setGradState(True)

        self.updateOptimizer(self.getLr())

        return True

    def trainStep(
        self,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
    ) -> Union[dict, None]:
        self.optimizer.zero_grad()

        boundary_pts, inner_pts = self.mash.toSamplePoints()[:2]

        fit_dists2, coverage_dists2 = mash_cpp.toChamferDistance(
            torch.vstack([boundary_pts, inner_pts]).unsqueeze(0), gt_points
        )[:2]

        fit_dists = torch.sqrt(fit_dists2 + EPSILON)
        coverage_dists = torch.sqrt(coverage_dists2 + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        boundary_connect_loss = torch.zeros_like(fit_loss)
        if boundary_connect_loss_weight > 0:
            boundary_connect_loss = mash_cpp.toBoundaryConnectLoss(
                self.mash.anchor_num, boundary_pts, self.mash.mask_boundary_phi_idxs
            )

        mean_fit_dist2 = torch.mean(fit_dists2)
        if mean_fit_dist2 >= self.max_fit_error2:
            self.fit_loss_weight *= self.fit_loss_scale
        else:
            self.fit_loss_weight /= self.fit_loss_scale
            self.fit_loss_weight = max(self.fit_loss_weight, 1.0)

        weighted_fit_loss = self.fit_loss_weight * fit_loss
        weighted_coverage_loss = coverage_loss_weight * coverage_loss
        weighted_boundary_connect_loss = (
            boundary_connect_loss_weight * boundary_connect_loss
        )

        loss = (
            weighted_fit_loss + weighted_coverage_loss + weighted_boundary_connect_loss
        )

        if torch.isnan(loss).any():
            print('[ERROR][BaseTrainer::trainStep]')
            print('\t loss is nan!')
            return None

        loss.backward()

        self.optimizer.step()

        self.coverage_percent = self.getCoveragePercent(coverage_dists2)

        self.addAnchor(coverage_dists2)

        loss_dict = {
            "State/boundary_pts": boundary_pts.shape[0],
            "State/inner_pts": inner_pts.shape[0],
            "Train/epoch": self.epoch,
            "Train/fit_loss": toNumpy(fit_loss),
            "Train/coverage_loss": toNumpy(coverage_loss),
            "Train/boundary_connect_loss": toNumpy(
                boundary_connect_loss
            ),
            "Train/weighted_fit_loss": toNumpy(weighted_fit_loss),
            "Train/weighted_coverage_loss": toNumpy(weighted_coverage_loss),
            "Train/weighted_boundary_connect_loss": toNumpy(
                weighted_boundary_connect_loss
            ),
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": toNumpy(fit_loss) + toNumpy(coverage_loss),
            "Metric/anchor_num": self.mash.anchor_num,
            "Metric/fit_loss_weight": self.fit_loss_weight,
            "Metric/coverage_percent": self.coverage_percent,
        }

        return loss_dict

    def autoTrainMash(
        self,
        gt_points_num: int = 400000,
    ) -> bool:
        boundary_connect_loss_weight_max = 0.0

        print("[INFO][AdaptiveTrainer::autoTrainMash]")
        print("\t start auto train Mash...")
        print(
            "\t degree: mask:",
            self.mash.mask_degree_max,
            "sh:",
            self.mash.sh_degree_max,
        )

        if self.mash.device == "cpu":
            gt_points_dtype = self.mash.dtype
        else:
            gt_points_dtype = torch.float32

        if self.gt_points is None:
            if not self.mesh.isValid():
                print("[ERROR][AdaptiveTrainer::autoTrainMash]")
                print("\t mesh is not valid!")
                return False

            self.gt_points = self.mesh.toSamplePoints(gt_points_num)

        gt_points = (
            torch.from_numpy(self.gt_points)
            .type(gt_points_dtype)
            .to(self.mash.device)
            .reshape(1, -1, 3)
        )

        while True:
            print("[INFO][AdaptiveTrainer::autoTrainMash]")
            print("\t start trainEpoch...")
            if not self.trainEpoch(
                self.lr,
                gt_points,
                1.0,
                0.5,
                0.0,
                self.warmup_step_num,
            ):
                print('[ERROR][AdaptiveTrainer::autoTrainMash]')
                print('\t trainEpoch failed!')
                return False

        print("[INFO][AdaptiveTrainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive loss...")
        for i in range(self.warmup_epoch):
            fit_loss_weight = 1.0

            manifold_loss_weight = i / (self.warmup_epoch - 1)

            coverage_loss_weight = 0.5 + 0.5 * manifold_loss_weight
            boundary_connect_loss_weight = (
                0.1 + 0.9 * manifold_loss_weight
            ) * boundary_connect_loss_weight_max

            if not self.trainEpoch(
                self.lr,
                gt_points,
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            ):
                print('[ERROR][AdaptiveTrainer::autoTrainMash]')
                print('\t trainEpoch failed!')
                return False

        print("[INFO][AdaptiveTrainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive lr...")
        current_lr = self.lr
        current_finetune_epoch = 1

        while current_lr > self.min_lr:
            current_lr *= self.factor
            if current_lr < self.min_lr:
                current_lr = self.min_lr

            current_finetune_epoch += 1

            fit_loss_weight = 1.0
            coverage_loss_weight = 1.0

            manifold_loss_weight = 1.0 / current_finetune_epoch

            boundary_connect_loss_weight = (
                0.0 + 1.0 * manifold_loss_weight
            ) * boundary_connect_loss_weight_max

            self.trainEpoch(
                current_lr,
                gt_points,
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            )

            if current_lr == self.min_lr:
                break

        self.autoSavePcd('final', add_idx=False)
        self.autoSaveMash('final')

        """
            if self.upperSHDegree():
                print("[INFO][AdaptiveTrainer::autoTrainMash]")
                print("\t upperSHDegree success!")
                print("\t start auto train Mash...")
                print(
                    "\t degree: mask:",
                    self.mash.mask_degree_max,
                    "sh:",
                    self.mash.sh_degree_max,
                )
                continue

            if self.upperMaskDegree():
                print("[INFO][AdaptiveTrainer::autoTrainMash]")
                print("\t upperMaskDegree success!")
                print("\t start auto train Mash...")
                print(
                    "\t degree: mask:",
                    self.mash.mask_degree_max,
                    "sh:",
                    self.mash.sh_degree_max,
                )
                continue

            break
        """

        return True
