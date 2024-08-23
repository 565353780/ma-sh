import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from copy import deepcopy
from torch.optim import AdamW

import mash_cpp

from ma_sh.Config.weights import W0
from ma_sh.Config.constant import EPSILON
from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Data.mesh import Mesh
from ma_sh.Method.data import toNumpy
from ma_sh.Method.path import removeFile
from ma_sh.Method.pcd import getPointCloud, downSample
from ma_sh.Method.time import getCurrentTime
from ma_sh.Model.mash import Mash
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer

mode = 'mash'

class Trainer(object):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        mask_boundary_sample_num: int = 90,
        sample_polar_num: int = 1000,
        sample_point_scale: float = 0.8,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cuda",
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        warmup_epoch: int = 4,
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
        self.warmup_step_num = warmup_step_num
        self.warmup_epoch = warmup_epoch
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

        self.mesh = Mesh()
        self.gt_points = None

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

        self.gt_points = np.asarray(sample_gt_pcd.points)

        center = np.mean(self.gt_points, axis=0)
        min_bound = np.min(self.gt_points, axis=0)
        max_bound = np.max(self.gt_points, axis=0)
        length = max_bound - min_bound
        self.translate = center
        self.scale = np.max(length)

        self.gt_points = (self.gt_points - self.translate) / self.scale

        sample_gt_pcd.estimate_normals()
        #sample_gt_pcd.orient_normals_consistent_tangent_plane(4)

        surface_dist = 0.001

        anchor_pcd = downSample(sample_gt_pcd, self.mash.anchor_num)

        if anchor_pcd is None:
            print("[ERROR][Trainer::loadGTPointsFile]")
            print("\t downSample failed!")
            return False

        sample_pts = np.asarray(anchor_pcd.points)
        sample_normals = np.asarray(anchor_pcd.normals)

        sh_params = torch.ones_like(self.mash.sh_params) * EPSILON
        sh_params[:, 0] = surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=sample_pts + surface_dist * sample_normals,
            face_forward_vectors=-sample_normals,
        )
        return True

    def loadGTPointsFile(self, gt_points_file_path: str, sample_point_num: Union[int, None] = None) -> bool:
        if not os.path.exists(gt_points_file_path):
            print("[ERROR][Trainer::loadGTPointsFile]")
            print("\t gt points file not exist!")
            print("\t gt_points_file_path:", gt_points_file_path)
            return False

        gt_points_file_type = gt_points_file_path.split('.')[-1]
        if gt_points_file_type == 'npy':
            gt_points = np.load(gt_points_file_path)
        else:
            gt_pcd = o3d.io.read_point_cloud(gt_points_file_path)
            gt_points = np.asarray(gt_pcd.points)

        if not self.loadGTPoints(gt_points, sample_point_num):
            return False

        return True

    def loadMeshFile(self, mesh_file_path: str) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][Trainer::loadMeshFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        if not self.mesh.loadMesh(mesh_file_path):
            print("[ERROR][Trainer::loadMeshFile]")
            print("\t loadMesh failed!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        surface_dist = 0.001

        self.mesh.samplePoints(self.mash.anchor_num)

        assert self.mesh.sample_normals is not None
        assert self.mesh.sample_pts is not None

        sh_params = torch.zeros_like(self.mash.sh_params)
        sh_params[:, 0] = surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=self.mesh.sample_pts + surface_dist * self.mesh.sample_normals,
            face_forward_vectors=-self.mesh.sample_normals,
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

    def upperMaskDegree(self) -> bool:
        if self.mash.mask_degree_max == MAX_MASK_DEGREE:
            return False

        if not self.mash.updateMaskDegree(self.mash.mask_degree_max + 1):
            print("[ERROR][Trainer::upperMaskDegree]")
            print("\t updateMaskDegree failed!")
            return False
        return True

    def upperSHDegree(self) -> bool:
        if self.mash.sh_degree_max == MAX_SH_DEGREE:
            return False

        if not self.mash.updateSHDegree(self.mash.sh_degree_max + 1):
            print("[ERROR][Trainer::upperMaskDegree]")
            print("\t updateSHDegree failed!")
            return False
        return True

    def updateLr(self, lr: float) -> bool:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return True

    def trainStep(
        self,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
    ) -> Union[dict, None]:
        self.optimizer.zero_grad()

        boundary_pts, inner_pts, inner_idxs = self.mash.toSamplePoints()

        if False:
            anchor_fit_loss, anchor_coverage_loss = (
                mash_cpp.toAnchorChamferDistanceLoss(
                    self.mash.anchor_num,
                    boundary_pts,
                    inner_pts,
                    self.mash.mask_boundary_phi_idxs,
                    inner_idxs,
                    gt_points,
                )
            )

            with torch.no_grad():
                anchor_bounds = (
                    mash_cpp.toAnchorBounds(
                        self.mash.anchor_num,
                        boundary_pts,
                        inner_pts,
                        self.mash.mask_boundary_phi_idxs,
                        inner_idxs,
                    )
                    .detach()
                    .clone()
                )

            anchor_lengths = (
                torch.norm(anchor_bounds[:, 1] - anchor_bounds[:, 0], p=2, dim=1)
                + EPSILON
            )
            anchor_shape_weights = 1.0 / torch.pow(anchor_lengths, 0.1)
            anchor_shape_weights /= torch.max(anchor_shape_weights)

            fit_loss = torch.mean(anchor_fit_loss)
            coverage_loss = torch.mean(anchor_coverage_loss * anchor_shape_weights)

        fit_loss = torch.tensor(0.0).type(gt_points.dtype).to(gt_points.device)
        coverage_loss = torch.zeros_like(fit_loss)
        boundary_connect_loss = torch.zeros_like(fit_loss)

        if fit_loss_weight > 0 or coverage_loss_weight > 0:
            fit_loss, coverage_loss = mash_cpp.toChamferDistanceLoss(
                torch.vstack([boundary_pts, inner_pts]), gt_points
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
            "Train/epoch": self.epoch,
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
        epoch_lr: float,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        self.mash.setGradState(True)

        print("[INFO][Trainer::warmUpEpoch]")
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

        for i in range(train_step_max):
            if self.render:
                if self.step % self.render_freq == 0:
                    self.renderMash(gt_points)

            current_lr = epoch_lr * (i + 1.0) / train_step_max
            self.updateLr(current_lr)
            if self.logger.isValid():
                self.logger.addScalar("Train/lr", current_lr, self.step)

            loss_dict = self.trainStep(
                gt_points,
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

    def trainEpoch(
        self,
        epoch_lr: float,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        boundary_connect_loss_weight: float,
        train_step_max: Union[int, None] = None,
    ) -> bool:
        self.mash.setGradState(True)

        self.updateLr(epoch_lr)
        if self.logger.isValid():
            self.logger.addScalar("Train/lr", epoch_lr, self.step)

        print("[INFO][Trainer::trainEpoch]")
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
                    self.renderMash(gt_points)

            loss_dict = self.trainStep(
                gt_points,
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

    def autoTrainMash(
        self,
        gt_points_num: int = 400000,
    ) -> bool:
        boundary_connect_loss_weight_max = 0.1

        print("[INFO][Trainer::autoTrainMash]")
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
                print("[ERROR][Trainer::autoTrainMash]")
                print("\t mesh is not valid!")
                return False

            self.gt_points = self.mesh.toSamplePoints(gt_points_num)

        gt_points = (
            torch.from_numpy(self.gt_points)
            .type(gt_points_dtype)
            .to(self.mash.device)
            .reshape(1, -1, 3)
        )

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start warmUpEpoch...")
        self.warmUpEpoch(self.lr, gt_points, 1.0, 0.5, 0.0, self.warmup_step_num)

        print("[INFO][Trainer::autoTrainMash]")
        print("\t start trainEpoch with adaptive loss...")
        for i in range(self.warmup_epoch):
            fit_loss_weight = 1.0

            manifold_loss_weight = i / (self.warmup_epoch - 1)

            coverage_loss_weight = 0.5 + 0.5 * manifold_loss_weight
            boundary_connect_loss_weight = (
                0.1 + 0.9 * manifold_loss_weight
            ) * boundary_connect_loss_weight_max

            self.trainEpoch(
                self.lr,
                gt_points,
                fit_loss_weight,
                coverage_loss_weight,
                boundary_connect_loss_weight,
            )

        print("[INFO][Trainer::autoTrainMash]")
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
                print("[INFO][Trainer::autoTrainMash]")
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
                print("[INFO][Trainer::autoTrainMash]")
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

    def renderMash(self, gt_points: torch.Tensor) -> bool:
        if self.o3d_viewer is None:
            print("[ERROR][Trainer::renderMash]")
            print("\t o3d_viewer is None!")
            return False

        with torch.no_grad():
            self.o3d_viewer.clearGeometries()

            mesh_abb_length = 2.0 * self.mesh.toABBLength()
            if mesh_abb_length == 0:
                mesh_abb_length = 1.1

            gt_pcd = getPointCloud(toNumpy(gt_points.squeeze(0)))
            gt_pcd.translate([-mesh_abb_length, 0, 0])
            self.o3d_viewer.addGeometry(gt_pcd)

            boundary_pts, inner_pts, inner_idxs = self.mash.toSamplePoints()
            detect_points = torch.vstack([boundary_pts, inner_pts])
            detect_points = toNumpy(detect_points)
            pcd = getPointCloud(detect_points)
            self.o3d_viewer.addGeometry(pcd)

            # self.mesh.paintJetColorsByPoints(detect_points)
            mesh = self.mesh.toO3DMesh()
            mesh.translate([mesh_abb_length, 0, 0])
            self.o3d_viewer.addGeometry(mesh)

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
