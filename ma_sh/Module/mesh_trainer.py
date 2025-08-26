import os
import torch
import numpy as np
from time import time
from tqdm import tqdm
from typing import Union
from torch.optim import SGD

from mesh_cut.Module.average_mesh_cutter import AverageMeshCutter

from chamfer_distance.Module.chamfer_distances import ChamferDistances

from ma_sh.Config.constant import W0, EPSILON, MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Method.time import getCurrentTime
from ma_sh.Method.path import removeFile
from ma_sh.Model.simple_mash import SimpleMash as Mash
from ma_sh.Module.early_stopping import EarlyStopping
from ma_sh.Module.logger import Logger
from ma_sh.Module.o3d_viewer import O3DViewer


class MeshTrainer(object):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        use_inv: bool = True,
        dtype=torch.float32,
        device: str = "cuda",
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
        lr: float = 2e-3,
        min_lr: float = 1e-3,
        warmup_step_num: int = 80,
        factor: float = 0.8,
        patience: int = 2,
        render: bool = False,
        render_freq: int = 1,
        render_init_only: bool = False,
        save_freq: int = 1,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.factor = factor
        self.patience = patience

        self.epoch = 0
        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.render = render
        self.render_freq = render_freq
        self.render_init_only = render_init_only
        self.save_freq = save_freq

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.gt_points = None

        self.mash = Mash(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            use_inv,
            dtype,
            device,
            sample_phi_num,
            sample_theta_num,
        )

        self.optimizer = SGD(
            [
                self.mash.mask_params,
                self.mash.sh_params,
                self.mash.ortho_poses,
                self.mash.positions,
            ],
            lr=lr,
        )

        self.initRecords()

        self.o3d_viewer = None
        if self.render:
            self.o3d_viewer = O3DViewer()
            self.o3d_viewer.createWindow()

        self.min_lr_reach_time = 0

        # tmp
        self.surface_dist = 1e-4

        self.sample_mash_time = 0
        self.chamfer_loss_time = 0
        self.start_time = time()
        self.error = 0

        self.save_pcd = False
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
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadMeshFile(
        self,
        mesh_file_path: str,
        dist_max: float = 1.0 / 200,
        points_per_submesh: int = 8192,
    ) -> bool:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][MeshTrainer::loadMeshFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return False

        mesh_cutter = AverageMeshCutter(mesh_file_path, dist_max)
        mesh_cutter.cutMesh(self.mash.anchor_num, points_per_submesh)

        if self.render:
            print("Render face labels...")
            mesh_cutter.renderFaceLabels()

            # print("Render sub meshes...")
            # mesh_cutter.renderSubMeshSamplePoints()

        self.gt_points = mesh_cutter.sub_mesh_sample_points

        mesh = mesh_cutter.toO3DMesh()
        mesh.compute_vertex_normals()

        fps_positions = np.asarray(mesh.vertices)[mesh_cutter.fps_vertex_idxs]

        fps_normals = np.asarray(mesh.vertex_normals)[mesh_cutter.fps_vertex_idxs]

        sh_params = torch.zeros_like(self.mash.sh_params)
        sh_params[:, 0] = self.surface_dist / W0

        self.mash.loadParams(
            sh_params=sh_params,
            positions=fps_positions + self.surface_dist * fps_normals,
            face_forward_vectors=-fps_normals,
        )
        return True

    def loadParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray],
        sh_params: Union[torch.Tensor, np.ndarray],
        ortho_poses: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
    ) -> bool:
        self.mash.loadParams(mask_params, sh_params, ortho_poses, positions)
        return True

    def loadBestParams(
        self,
        mask_params: Union[torch.Tensor, np.ndarray],
        sh_params: Union[torch.Tensor, np.ndarray],
        ortho_poses: Union[torch.Tensor, np.ndarray],
        positions: Union[torch.Tensor, np.ndarray],
    ) -> bool:
        if isinstance(mask_params, np.ndarray):
            mask_params = torch.from_numpy(mask_params)
        if isinstance(sh_params, np.ndarray):
            sh_params = torch.from_numpy(sh_params)
        if isinstance(ortho_poses, np.ndarray):
            ortho_poses = torch.from_numpy(ortho_poses)
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions)

        self.best_params_dict["mask_params"] = mask_params
        self.best_params_dict["sh_params"] = sh_params
        self.best_params_dict["ortho_poses"] = ortho_poses
        self.best_params_dict["positions"] = positions
        return True

    def upperMaskDegree(self) -> bool:
        if self.mash.mask_degree_max == MAX_MASK_DEGREE:
            return False

        if not self.mash.updateMaskDegree(self.mash.mask_degree_max + 1):
            print("[ERROR][BaseTrainer::upperMaskDegree]")
            print("\t updateMaskDegree failed!")
            return False
        return True

    def upperSHDegree(self) -> bool:
        if self.mash.sh_degree_max == MAX_SH_DEGREE:
            return False

        if not self.mash.updateSHDegree(self.mash.sh_degree_max + 1):
            print("[ERROR][BaseTrainer::upperMaskDegree]")
            print("\t updateSHDegree failed!")
            return False
        return True

    def getLr(self) -> bool:
        return self.optimizer.param_groups[0]["lr"]

    def updateLr(self, lr: float) -> bool:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return True

    def trainStep(
        self,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
    ) -> Union[dict, None]:
        self.optimizer.zero_grad()

        start = time()
        mash_pts = self.mash.toSamplePoints().reshape(self.mash.anchor_num, -1, 3)
        self.sample_mash_time += time() - start

        start = time()
        fit_loss = torch.tensor(0.0).type(gt_points.dtype).to(gt_points.device)
        coverage_loss = torch.zeros_like(fit_loss)

        if torch.cuda.is_available() and gt_points.device != "cpu":
            fit_dists2, coverage_dists2 = ChamferDistances.namedAlgo("cuda")(
                mash_pts, gt_points
            )[:2]
        else:
            fit_dists2, coverage_dists2 = ChamferDistances.namedAlgo("default")(
                mash_pts, gt_points
            )[:2]

        fit_dists = torch.sqrt(fit_dists2 + EPSILON)
        coverage_dists = torch.sqrt(coverage_dists2 + EPSILON)

        fit_loss = torch.mean(fit_dists)
        coverage_loss = torch.mean(coverage_dists)

        spend = time() - start
        self.chamfer_loss_time += spend

        weighted_fit_loss = fit_loss_weight * fit_loss
        weighted_coverage_loss = coverage_loss_weight * coverage_loss

        loss = weighted_fit_loss + weighted_coverage_loss

        if torch.isnan(loss).any():
            print("[ERROR][BaseTrainer::trainStep]")
            print("\t loss is nan!")
            print("\t\t fit_loss:", fit_loss)
            print("\t\t coverage_loss:", coverage_loss)
            return None

        loss.backward()

        self.mash.clearNanGrads()

        self.optimizer.step()

        chamfer_distance = toNumpy(fit_loss) + toNumpy(coverage_loss)
        self.error = chamfer_distance

        loss_dict = {
            "Train/epoch": self.epoch,
            "Train/fit_loss": toNumpy(fit_loss),
            "Train/coverage_loss": toNumpy(coverage_loss),
            "Train/weighted_fit_loss": toNumpy(weighted_fit_loss),
            "Train/weighted_coverage_loss": toNumpy(weighted_coverage_loss),
            "Train/loss": toNumpy(loss),
            "Metric/chamfer_distance": chamfer_distance,
        }

        return loss_dict

    def warmUpEpoch(
        self,
        epoch_lr: float,
        gt_points: torch.Tensor,
        fit_loss_weight: float,
        coverage_loss_weight: float,
        train_step_max: int,
    ) -> bool:
        self.mash.setGradState(True)

        print("[INFO][BaseTrainer::warmUpEpoch]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
        pbar = tqdm(total=train_step_max)

        start_step = self.step

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
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::warmUpEpoch]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            for key, item in loss_dict.items():
                self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSavePcd("train", self.save_freq, False)
            self.autoSaveMash("train", self.save_freq)

            self.step += 1
            pbar.update(1)

            self.logger.addScalar("Train/patience", self.patience, self.step)

            if self.step >= start_step + train_step_max:
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
        train_step_max: Union[int, None] = None,
    ) -> bool:
        early_stopping = EarlyStopping(self.patience, 1e-6)

        self.mash.setGradState(True)

        self.updateLr(epoch_lr)
        if self.logger.isValid():
            self.logger.addScalar("Train/lr", epoch_lr, self.step)

        print("[INFO][BaseTrainer::trainEpoch]")
        print("\t start train epoch :", self.epoch, "with lr :", epoch_lr, "...")
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
            )

            if loss_dict is None:
                print("[ERROR][BaseTrainer::trainEpoch]")
                print("\t trainStep failed!")
                return False

            assert isinstance(loss_dict, dict)
            for key, item in loss_dict.items():
                self.logger.addScalar(key, item, self.step)

            loss = loss_dict["Train/loss"]

            pbar.set_description("LOSS %.6f" % (loss,))

            self.autoSavePcd("train", self.save_freq, False)
            self.autoSaveMash("train", self.save_freq)

            self.step += 1
            pbar.update(1)

            if train_step_max is not None:
                self.logger.addScalar("Train/patience", self.patience, self.step)

                if self.step >= start_step + train_step_max:
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

            if early_stopping(loss.item()):
                break

        self.logger.addScalar("Train/lr", epoch_lr, self.step - 1)

        self.epoch += 1
        return True

    def autoTrainMash(self) -> bool:
        print("[INFO][MeshTrainer::autoTrainMash]")
        print("\t start auto train SimpleMash...")
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

        gt_points = self.gt_points.to(self.mash.device, dtype=gt_points_dtype)

        print("[INFO][MeshTrainer::autoTrainMash]")
        print("\t start warmUpEpoch...")
        if not self.warmUpEpoch(self.lr, gt_points, 1.0, 0.5, self.warmup_step_num):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t warmUpEpoch failed!")
            return False

        if not self.trainEpoch(self.lr, gt_points, 1.0, 0.5):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t trainEpoch failed!")
            return False

        if not self.warmUpEpoch(self.lr, gt_points, 1.0, 1.0, self.warmup_step_num):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t warmUpEpoch failed!")
            return False

        if not self.trainEpoch(self.lr, gt_points, 1.0, 1.0):
            print("[ERROR][Trainer::autoTrainMash]")
            print("\t trainEpoch failed!")
            return False

        self.autoSavePcd("final", add_idx=False)
        self.autoSaveMash("final")

        total_time = time() - self.start_time

        print("[INFO][Trainer::autoTrainMash]")
        print("\t training finished! metrics:")
        print("\t surface sampling:", self.sample_mash_time)
        print("\t chamfer loss:", self.chamfer_loss_time, "s")
        print("\t total:", total_time)
        print("\t error:", self.error)

        if self.o3d_viewer is not None:
            self.o3d_viewer.run()

        return True

    def saveMashFile(self, save_mash_file_path: str, overwrite: bool = False) -> bool:
        sample_mash = self.mash.clone()

        sample_mash.saveParamsFile(save_mash_file_path, overwrite)
        return True

    def autoSaveMash(
        self, state_info: str, save_freq: int = 1, add_idx: bool = True
    ) -> bool:
        if self.save_result_folder_path is None:
            return False

        if save_freq <= 0:
            return False

        if self.save_file_idx % save_freq != 0:
            if add_idx:
                self.save_file_idx += 1
            return False

        save_file_path = (
            self.save_result_folder_path
            + "mash/"
            + str(self.save_file_idx)
            + "_"
            + state_info
            + "_anc-"
            + str(self.mash.anchor_num)
            + "_mash.npy"
        )

        if not self.saveMashFile(save_file_path, True):
            print("[ERROR][BaseTrainer::autoSaveMash]")
            print("\t saveMashFile failed!")
            return False

        if add_idx:
            self.save_file_idx += 1
        return True

    def saveAsPcdFile(self, save_pcd_file_path: str, overwrite: bool = False) -> bool:
        if os.path.exists(save_pcd_file_path):
            if not overwrite:
                return True

            removeFile(save_pcd_file_path)

        save_mash = self.mash.clone()

        save_mash.saveAsPcdFile(save_pcd_file_path, overwrite=overwrite)
        return True

    def autoSavePcd(
        self, state_info: str, save_freq: int = 1, add_idx: bool = True
    ) -> bool:
        if not self.save_pcd:
            return False

        if self.save_result_folder_path is None:
            return False

        if save_freq <= 0:
            return False

        if self.save_file_idx % save_freq != 0:
            if add_idx:
                self.save_file_idx += 1
            return False

        save_pcd_file_path = (
            self.save_result_folder_path
            + "pcd/"
            + str(self.save_file_idx)
            + "_"
            + state_info
            + "_pcd.ply"
        )

        if not self.saveAsPcdFile(save_pcd_file_path, overwrite=True):
            print("[ERROR][BaseTrainer::autoSavePcd]")
            print("\t saveAsPcdFile failed!")
            return False

        if add_idx:
            self.save_file_idx += 1
        return True

    def renderMash(self, gt_points: torch.Tensor) -> bool:
        if self.o3d_viewer is None:
            print("[ERROR][BaseTrainer::renderMash]")
            print("\t o3d_viewer is None!")
            return False

        with torch.no_grad():
            self.o3d_viewer.clearGeometries()

            # gt_pcd = getPointCloud(toNumpy(gt_points.squeeze(0)))
            # self.o3d_viewer.addGeometry(gt_pcd)

            detect_points = self.mash.toSamplePoints()
            pcd = getPointCloud(toNumpy(detect_points))
            self.o3d_viewer.addGeometry(pcd)

            """
            for j in range(self.mash.mask_params.shape[0]):
                view_cone = self.toO3DViewCone(j)
                self.o3d_viewer.addGeometry(view_cone)

                # inv_sphere = self.toO3DInvSphere(j)
                # inv_sphere.translate([-30, 0, 0])
                # self.o3d_viewer.addGeometry(inv_sphere)
            """

            self.o3d_viewer.update()

            if self.render_init_only:
                self.o3d_viewer.run()
                # exit()
        return True
