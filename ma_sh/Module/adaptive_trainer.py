import os
import torch
import numpy as np
import open3d as o3d
from typing import Union
from torch.optim import AdamW

from ma_sh.Config.weights import W0
from ma_sh.Config.constant import EPSILON
from ma_sh.Method.pcd import downSample, toMergedPcd
import mash_cpp

from ma_sh.Model.mash import Mash
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.path import createFileFolder, removeFile, renameFile
from ma_sh.Module.base_trainer import BaseTrainer

mode = 'mash'

class AdaptiveTrainer(BaseTrainer):
    def __init__(
        self,
        init_anchor_num: int = 100,
        add_anchor_num: int = 50,
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
        refine_step_num: int = 20,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = 1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.add_anchor_num = add_anchor_num
        self.max_fit_error = max_fit_error
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
        self.factor = factor
        self.refine_step_num = refine_step_num

        self.max_fit_error2 = self.max_fit_error * self.max_fit_error
        self.distance_thresh = max(self.max_fit_error2, 1e-4)

        self.not_fit_loss_weight = 1000.0

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

        self.merged_mash = None

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

        self.coverage_percent = 0
        self.coverage_all_surface = False
        self.coverage_dists2 = torch.empty([0], dtype=self.mash.dtype, device=self.mash.device)
        return

    def updateOptimizer(self, lr: float) -> bool:
        self.mash.setGradState(True)
        self.mash.clearGrads()

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
        coveraged_num = (coverage_dists2 <= self.distance_thresh).sum().item()
        total_elements = coverage_dists2.numel()
        coverage_percent = 100.0 * coveraged_num / total_elements

        if coverage_percent >= 99.8:
            self.coverage_all_surface = True

        return coverage_percent

    @torch.no_grad()
    def addAnchor(self) -> bool:
        not_fitted_point_idxs = self.coverage_dists2.cpu().flatten() > self.distance_thresh

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

        if self.merged_mash is None:
            self.merged_mash = self.mash.clone()
        else:
            self.merged_mash.mergeMash(self.mash)

            self.merged_mash.clearGrads()
            self.merged_mash.setGradState(False)

        self.mash = Mash.fromMash(self.merged_mash, anchor_num=sample_pts.shape[0])

        new_sh_params = torch.ones([sample_pts.shape[0], self.mash.sh_params.shape[1]],
                                   dtype=self.mash.dtype, device=self.mash.device) * EPSILON
        new_sh_params[:, 0] = self.surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=new_sh_params,
            positions=sample_pts + self.surface_dist * sample_normals,
            face_forward_vectors=-sample_normals
        )

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

        if self.merged_mash is None:
            sample_pts = torch.vstack([boundary_pts, inner_pts])
        else:
            merged_boundary_pts, merged_inner_pts = self.merged_mash.toSamplePoints()[:2]
            sample_pts = torch.vstack([boundary_pts, inner_pts, merged_boundary_pts, merged_inner_pts])

        fit_dists2, coverage_dists2, _, coverage_idxs = mash_cpp.toChamferDistance(
            sample_pts.unsqueeze(0), gt_points
        )

        valid_idx_max = boundary_pts.shape[0] + inner_pts.shape[0]

        valid_fit_dists2 = fit_dists2[:, :valid_idx_max]
        valid_coverage_dists2 = coverage_dists2[coverage_idxs < valid_idx_max]

        self.coverage_dists2 = coverage_dists2.detach().clone()

        fit_dists = torch.sqrt(valid_fit_dists2 + EPSILON)
        coverage_dists = torch.sqrt(valid_coverage_dists2 + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        boundary_connect_loss = torch.zeros_like(fit_loss)
        if boundary_connect_loss_weight > 0:
            boundary_connect_loss = mash_cpp.toBoundaryConnectLoss(
                self.mash.anchor_num, boundary_pts, self.mash.mask_boundary_phi_idxs
            )

        not_fit_loss = torch.zeros_like(fit_loss)
        if self.not_fit_loss_weight > 0:
            not_fit_dists2 = valid_fit_dists2[valid_fit_dists2 > self.max_fit_error2]
            if not_fit_dists2.shape[0] > 0:
                not_fit_loss = torch.mean(not_fit_dists2) - self.max_fit_error2

        weighted_not_fit_loss = self.not_fit_loss_weight * not_fit_loss
        weighted_fit_loss = fit_loss_weight * fit_loss
        weighted_coverage_loss = coverage_loss_weight * coverage_loss
        weighted_boundary_connect_loss = (
            boundary_connect_loss_weight * boundary_connect_loss
        )

        loss = (
            weighted_not_fit_loss + weighted_fit_loss + weighted_coverage_loss + weighted_boundary_connect_loss
        )

        if torch.isnan(loss).any():
            print('[ERROR][BaseTrainer::trainStep]')
            print('\t loss is nan!')
            return None

        loss.backward()

        self.optimizer.step()

        self.coverage_percent = self.getCoveragePercent(coverage_dists2)

        boundary_pts_num = boundary_pts.shape[0]
        inner_pts_num = inner_pts.shape[0]
        anchor_num = self.mash.anchor_num
        if self.merged_mash is not None:
            boundary_pts_num += merged_boundary_pts.shape[0]
            inner_pts_num += merged_inner_pts.shape[0]
            anchor_num += self.merged_mash.anchor_num

        loss_dict = {
            "State/boundary_pts": boundary_pts_num,
            "State/inner_pts": inner_pts_num,
            "Train/epoch": self.epoch,
            "Train/not_fit_loss": toNumpy(not_fit_loss),
            "Train/fit_loss": toNumpy(fit_loss),
            "Train/coverage_loss": toNumpy(coverage_loss),
            "Train/boundary_connect_loss": toNumpy(
                boundary_connect_loss
            ),
            "Train/weighted_not_fit_loss": toNumpy(weighted_not_fit_loss),
            "Train/weighted_fit_loss": toNumpy(weighted_fit_loss),
            "Train/weighted_coverage_loss": toNumpy(weighted_coverage_loss),
            "Train/weighted_boundary_connect_loss": toNumpy(
                weighted_boundary_connect_loss
            ),
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": toNumpy(fit_loss) + toNumpy(coverage_loss),
            "Metric/anchor_num": anchor_num,
            "Metric/coverage_percent": self.coverage_percent,
        }

        return loss_dict

    def refineMergedMash(self,
                         gt_points: torch.Tensor,
                         refine_step_num: Union[int, None] = None) -> bool:
        boundary_connect_loss_weight_max = 0.1

        source_patience = self.patience

        self.patience = 2

        if self.merged_mash is not None:
            self.mash.mergeMash(self.merged_mash)
            self.merged_mash = None

        self.updateOptimizer(self.getLr())

        print("[INFO][AdaptiveTrainer::refineMergedMash]")
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
                refine_step_num,
            ):
                print('[ERROR][AdaptiveTrainer::refineMergedMash]')
                print('\t trainEpoch failed!')
                return False

        print("[INFO][AdaptiveTrainer::refineMergedMash]")
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
                refine_step_num,
            )

            if current_lr == self.min_lr:
                break

        self.patience = source_patience

        return True

    def autoTrainMash(
        self,
        gt_points_num: int = 400000,
    ) -> bool:
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

        print("[INFO][AdaptiveTrainer::autoTrainMash]")
        print("\t start trainEpoch to coverage full shape...")
        if not self.trainEpoch(
            self.lr,
            gt_points,
            1.0,
            0.5,
            0.0,
        ):
            print('[ERROR][AdaptiveTrainer::autoTrainMash]')
            print('\t trainEpoch failed!')
            return False

        while not self.coverage_all_surface:
            self.addAnchor()

            print("[INFO][AdaptiveTrainer::autoTrainMash]")
            print("\t start trainEpoch to coverage full shape...")
            if not self.trainEpoch(
                self.lr,
                gt_points,
                1.0,
                0.5,
                0.0,
            ):
                print('[ERROR][AdaptiveTrainer::autoTrainMash]')
                print('\t trainEpoch failed!')
                return False

            if self.coverage_all_surface:
                break

            if not self.refineMergedMash(
                gt_points,
                self.refine_step_num,
            ):
                print('[ERROR][AdaptiveTrainer::autoTrainMash]')
                print('\t refineMergedMash failed!')
                return False

            if self.coverage_all_surface:
                break

        if not self.refineMergedMash(gt_points):
            print('[ERROR][AdaptiveTrainer::autoTrainMash]')
            print('\t refineMergedMash failed for the last call!')
            return False

        self.autoSavePcd('final', add_idx=False)
        self.autoSaveMash('final')

        if self.o3d_viewer is not None:
            self.o3d_viewer.run()

        return True


    def saveAsPcdFile(self, save_pcd_file_path: str, overwrite: bool = False) -> bool:
        print_progress = False

        if os.path.exists(save_pcd_file_path):
            if not overwrite:
                return True

            removeFile(save_pcd_file_path)

        save_mash = self.mash.clone()

        if self.translate is not None:
            save_mash.scale(1.0 / self.scale, False)
            save_mash.translate(self.translate)

        save_pcd = save_mash.toSamplePcd()

        if self.merged_mash is not None:
            save_merged_mash = self.merged_mash.clone()

            if self.translate is not None:
                save_merged_mash.scale(1.0 / self.scale, False)
                save_merged_mash.translate(self.translate)

            save_merged_pcd = save_merged_mash.toSamplePcd()

            save_pcd = toMergedPcd(save_pcd, save_merged_pcd)

        createFileFolder(save_pcd_file_path)

        tmp_save_pcd_file_path = save_pcd_file_path[:-4] + "_tmp" + save_pcd_file_path[-4:]

        o3d.io.write_point_cloud(
            tmp_save_pcd_file_path, save_pcd, write_ascii=True, print_progress=print_progress
        )

        renameFile(tmp_save_pcd_file_path, save_pcd_file_path)

        return True
